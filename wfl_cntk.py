import cntk
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import psutil
import sys
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import sessionmaker

DeclarativeBase = declarative_base()

class Model(DeclarativeBase):
    __tablename__ = 'models'

    id = Column(Integer, primary_key = True)
    name = Column(String, nullable = False)
    __table_args__ = (UniqueConstraint('name', name='unco1'),)

class Forecast(DeclarativeBase):
    __tablename__ = 'forecasts'

    id = Column(Integer, primary_key = True)
    model = Column(Integer, ForeignKey('models.id'), nullable = False)
    symbol = Column(String)
    ts = Column(DateTime)
    fore = Column(Float)
    details = Column(String)
    __table_args__ = (UniqueConstraint('model', 'ts', 'symbol', name='unco1'),)

class ForecastLocations:
    def __init__(self, timestamps, nahead=1, min_history=1000, max_history=1e6, start_index=None, end_index=None, start_date=None, end_date=None, history_delay=0):
        self.starts = None
        self.ends = None
        
        timestamps = np.array(timestamps, dtype='datetime64')

        ll = len(timestamps)

        if start_index is None:
            if start_date is not None:
                start_date = np.datetime64(start_date)
                start_index = max(np.searchsorted(timestamps, start_date), min_history + 1 + history_delay)
            else:
                start_index = min_history + history_delay

        if end_index is None:
            if end_date is None:
                end_index = ll - 1
            else:
                end_date = np.datetime64(end_date)
                end_index = np.searchsorted(timestamps, end_date)

        if start_index >= end_index:
            return

        if len(np.unique(timestamps)) == ll:
            self.ends = np.arange(start=nahead, stop=end_index - start_index + 1 + 1, step=nahead) + start_index - 1
            self.starts = self.ends - nahead + 1
        else:
            # diffs = np.ndarray.astype(np.ediff1d(timestamps, to_begin=np.timedelta64(1)), 'int')
            diffs = np.ndarray.astype(np.ediff1d(timestamps), 'int64')
            starts = np.arange(len(diffs))[diffs != 0]
            self.starts = starts[(starts >= start_index) & (starts <= end_index)]
            self.ends = np.roll(self.starts, -1) - 1
            self.ends[-1] = end_index

    def len(self):
        if self.starts is None:
            return None
        return len(self.starts)

class WalkForwardLoop:
    def __init__(self, model_name, log_file, classifier=None, index_format='%Y-%m-%d', db_url=None, scale=True, verbose=False):
        self.model_name = model_name # The model name to use for the database
        self.classifier = classifier # The classifier object
        self.log_file = log_file
        self.index_format = index_format

        self.db_url = db_url
        self.db_session = None

        self.scale = scale

        self.verbose = verbose

    def init_db(self):
        engine = create_engine(self.db_url)
        DeclarativeBase.metadata.create_all(engine)
        Session = sessionmaker(bind = engine)
        self.db_session = Session()
        try:
            self.db_session.add(Model(name = self.model_name))
            self.db_session.commit()
        except:
            self.db_session.rollback()
            pass
        self.model_id = self.db_session.query(Model.id).filter(Model.name == self.model_name).first()[0]

    def run_step(self, id, lock=None):
        if self.db_url is not None:
            self.init_db()

        # Prepare the range for training for this iteration
        history_end = self.forecast_locations.starts[id]
        history_start = 0
        if (history_end - history_start + 1) > self.max_history:
            history_start = history_end - max_history + 1 
        xx = self.features.iloc[history_start:history_end].as_matrix()
        yy = self.response.iloc[history_start:history_end].as_matrix()

        # Scale the data
        if self.scale:
            std_scaler = StandardScaler()
            xx = std_scaler.fit_transform(xx)

        fore_xx = self.features.iloc[self.forecast_locations.starts[id]:(self.forecast_locations.ends[id]+1)].as_matrix()
        if self.scale:
            fore_xx = std_scaler.transform(fore_xx)

        if sys.platform == 'win32':
            timer = time.clock
        else:
            timer = time.time

        # Train the model and predict
        start = timer()
        fore = self.cntk_fit_predict(xx, yy, fore_xx)
        forecasting_time = timer() - start

        fore_df = pd.DataFrame(fore, index=self.features.iloc[self.forecast_locations.starts[id]:(self.forecast_locations.ends[id]+1)].index)
        # Generate proper column names. Map -1,0,1 to 'short','out','long'. The 4th column is the class.
        # fore_df.columns = np.append(np.array(['short','long'])[self.classes.astype(int) + 1], ['class'])
        fore_df.ix[:,2] = np.where(fore_df.ix[:,2] == -1, 'short', 'long')
        fore_df.columns = np.array(['short_prob', 'long_prob', 'class'])

        # print(fore_df)

        fore = fore[:,2]

        metric = np.round(np.amax(fore_df.ix[:,0:4], axis=1), 2)

        if lock is not None:
            lock.acquire()

        try:
            # Save results to a database or somewhere else
            if self.db_session is not None:
                for jj in range(len(fore)):
                    row_id = self.forecast_locations.starts[id] + jj
                    ts = self.features.index[row_id]
                    details = fore_df.iloc[[jj]].to_json(orient='split', date_format='iso')
                    if self.symbol_column is not None:
                        symbol = self.symbol_column[row_id]
                        rs = self.db_session.query(Forecast.id).filter(Forecast.ts == ts).filter(Forecast.model == self.model_id).filter(Forecast.symbol == symbol).first()
                        if rs is None:
                            ff = Forecast(model = self.model_id, ts = ts, fore = fore[jj], details = details, symbol = symbol)
                            self.db_session.add(ff)
                        else:
                            ff = Forecast(id = rs[0], model = self.model_id, ts = ts, fore = fore[jj], details = details, symbol = symbol)
                            self.db_session.merge(ff)
                    else:
                        rs = self.db_session.query(Forecast.id).filter(Forecast.ts == ts).filter(Forecast.model == self.model_id).first()
                        if rs is None:
                            ff = Forecast(model = self.model_id, ts = ts, fore = fore[jj], details = details)
                            self.db_session.add(ff)
                        else:
                            ff = Forecast(id = rs[0], model = self.model_id, ts = ts, fore = fore[jj], details = details)
                            self.db_session.merge(ff)

            # Log output
            if self.log_file is not None:
                out_str = "\n" + self.features.index[self.forecast_locations.starts[id]].strftime(self.index_format) + " - " + \
                    self.features.index[self.forecast_locations.ends[id]].strftime(self.index_format) + "\n" + \
                    "=======================\n" + \
                    "    history: from: " + self.features.index[history_start].strftime(self.index_format) + ", to: " + \
                    self.features.index[history_end - 1].strftime(self.index_format) + \
                    ", length: " + str(history_end - history_start) + "\n" + \
                    "    forecast length: " + str(self.forecast_locations.ends[id] - self.forecast_locations.starts[id] + 1) + "\n" + \
                    "    forecast: [" + ','.join(str(round(ff, 2)) for ff in fore) + "]\n" + \
                    "    probs: [" + ','.join(str(round(mm, 2)) for mm in metric) + "]\n" + \
                    "    time [training+forecasting]: " + str(round(forecasting_time, 2)) + " secs\n"
                with open(self.log_file, "a") as ff:
                    print(out_str, file=ff)
        finally:
            if lock is not None:
                lock.release()

        if self.db_session is not None:
            self.db_session.commit()

    def run(self, features, response, forecast_locations, max_history=1e6, symbol_column=None, verbose=None, pool_size=1):
        assert len(features) == len(response)

        if isinstance(verbose, bool):
            self.verbose = verbose

        self.forecast_locations = forecast_locations
        self.features = features
        self.response = response
        self.max_history = max_history
        self.symbol_column = symbol_column
        self.verbose = verbose

        self.lock = None

        if pool_size > 1:
            lock = mp.Lock()
            pool = mp.pool.Pool(pool_size, initializer=lock_init, initargs=(lock,))
            for id in range(0, forecast_locations.len()):
                pool.apply_async(apply_step, args=(self, id))

            pool.close()
            pool.join()
        else:
            for ii in range(0, forecast_locations.len()):
                self.run_step(ii)

    def print_training_progress(self, trainer, mb, frequency):
        training_loss = "NA"
        eval_error = "NA"
        if mb % frequency == 0:
            training_loss = get_train_loss(trainer)
            eval_error = get_train_eval_criterion(trainer)
            if self.verbose:
                logging.info("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb + 1, training_loss, eval_error * 100))
            return mb, training_loss, eval_error

    def cntk_fit_predict(self, x, y, newx):
        learning_rate = 0.01
        batch_size = 'auto'
        num_passes = 2
        display_step = 1

        if isinstance(batch_size, str):
            if batch_size == 'auto':
                batch_size = min(200, x.shape[0])
            else:
                raise ValueError("'auto' is the only acceptable string for batch_size")

        num_batches = x.shape[0] // batch_size

        # Map the y's to [0,nlevels)
        classes = np.sort(np.unique(y))
        self.classes = classes
        yz = np.searchsorted(classes, y)

        # One hot encode them
        ohe = OneHotEncoder(n_values=len(classes), sparse=False)
        yy = ohe.fit_transform(yz)

        # Build the classifier
        input = cntk.ops.input(x.shape[1], dtype=np.float32)
        label = cntk.ops.input(yy.shape[1], dtype=np.float32)

        #with cntk.default_options(dtype=np.float64):
        hh = cntk.layers.Sequential([
                cntk.layers.Convolution1D(3, 32, activation=cntk.ops.relu, pad=True, reduction_rank=0),
                cntk.layers.MaxPooling((3, 1), 3),
                cntk.layers.Convolution1D(3, 32, activation=cntk.ops.relu, pad=True),
                cntk.layers.MaxPooling((3, 1), 3),
                cntk.layers.Dense(128, activation=cntk.ops.relu),
                cntk.layers.Dense(128, activation=cntk.ops.relu),
                cntk.layers.Dense(yy.shape[1], activation=None)
                ])(input)

        loss = cntk.losses.cross_entropy_with_softmax(hh, label)
        label_error = cntk.metrics.classification_error(hh, label)
        lr_per_minibatch = cntk.learners.learning_rate_schedule(learning_rate, cntk.learners.UnitType.minibatch)
        trainer = cntk.Trainer(hh, (loss, label_error), [cntk.learners.sgd(hh.parameters, lr=lr_per_minibatch)])

        num_batches = x.shape[0] // batch_size

        res = None

        nfeatures = x.shape[1]
        nlabels = yy.shape[1]

        total_batches = num_batches * num_passes

        # Train our neural network
        tf = np.array_split(x, num_batches)
        tl = np.array_split(yy, num_batches)

        for ii in range(total_batches):
            features = np.ascontiguousarray(tf[ii % num_batches]).astype(np.float32)
            labels = np.ascontiguousarray(tl[ii % num_batches]).astype(np.float32)

            # Specify the mapping of input variables in the model to actual minibatch data to be trained with
            trainer.train_minibatch({input : features, label : labels})

        # Predict
        out = cntk.ops.softmax(hh)
        probs = np.squeeze(out.eval({input: newx.astype(np.float32)}))
        # Add a dimension if we squeezed too much
        if len(probs.shape) == 1:
            probs = np.reshape(probs, (1,-1))
        # Append the resulting class to the probabilities
        res = np.append(probs, [self.classes[np.argmax(probs, 1)]], axis=1)
        return(res)

def lock_init(lock):
    global global_lock
    global_lock = lock

def apply_step(wfl, id):
    wfl.run_step(id, lock=global_lock)
            
def returns_wfl():
    csv_path = 'c:/pprojects/tradingml/series.csv'
    ss = pd.read_csv(csv_path, header=None, parse_dates=True, index_col=0)

    rets = ss.pct_change()
    erets = rets.pow(2).ewm(span=36).mean().pow(1/2)
    arets = rets / erets
    arets = arets.dropna()

    history_len = 3*252 # Three years
    nrows = len(arets) - history_len
    mm = np.full((nrows, history_len), np.nan)
    for ii in range(history_len, len(arets)):
        mm[ii - history_len,:] = arets[(ii - history_len + 1):(ii + 1)].as_matrix().reshape((1,-1))

    response = np.where(arets < 0, -1, 1)
    response = pd.DataFrame(response, index=arets.index)
    # Remove the first history_len + 1. The extra one removed is
    # because we need to shift the features one position forward,
    # to align with the response, thus, we loose one more feature.
    response = response.tail(-history_len - 1)
    features = mm[:(mm.shape[0] - 1),:]
    features = pd.DataFrame(features, index=response.index)

    fl = ForecastLocations(features.index, start_date="2015-05-20")

    ml = WalkForwardLoop('cntk_conv_self', log_file='ml.log', db_url='sqlite:///ml.sqlite')
    ml.run(features, response, fl, verbose=False, pool_size=2)

def main():

    # Init logging
    logging.basicConfig(filename='diag.log',level=logging.DEBUG)

    returns_wfl()

if __name__ == "__main__":
    main()            # Save results to a database or somewhere else