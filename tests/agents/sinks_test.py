
import gc
import os
from random import random
import pandas as pd
import unittest

from dmas.models.trackers import DataSink
from dmas.utils.tools import print_scenario_banner


class TestDataSink(unittest.TestCase):
    def setUp(self):
        # define output directory and data sink parameters
        self.out_dir = "./tests/agents/.temp"
        self.owner_name = "test_agent"
        self.small_data_name = "small_test_data"
        self.large_data_name = "large_test_data"

        # make sure output directory exists
        os.makedirs(self.out_dir, exist_ok=True)

        # if there are any existing test output files, remove them
        for filename in os.listdir(self.out_dir):
            if filename.startswith(self.owner_name) and (filename.endswith(f"{self.small_data_name}.parquet") or filename.endswith(f"{self.large_data_name}.parquet")):
                os.remove(os.path.join(self.out_dir, filename))

        # generate some test data
        self.small_data = [
            {"col1": 1, "col2": "a"},
            {"col1": 2, "col2": "b"},
            {"col1": 3, "col2": "c"},
        ]
        self.large_data = [ {"col1": 10_000*random(), "col2": chr(97 + i)} 
                             for i in range(1000) ]

    def tearDown(self):
        # clean up any test output files
        for filename in os.listdir(self.out_dir):
            if filename.startswith(self.owner_name) and (filename.endswith(f"{self.small_data_name}.parquet") or filename.endswith(f"{self.large_data_name}.parquet")):
                os.remove(os.path.join(self.out_dir, filename))

        # remove the output directory if it is empty
        if not os.listdir(self.out_dir):
            os.rmdir(self.out_dir)

    def test_initializer(self):      
        # create a DataSink
        sink = DataSink(out_dir=self.out_dir, owner_name=self.owner_name, data_name=self.small_data_name, flush_rows=2)
        
        # check that attributes are set correctly
        self.assertEqual(sink.out_dir, self.out_dir)
        self.assertEqual(sink.owner_name, self.owner_name)
        self.assertEqual(sink.data_name, self.small_data_name)
        self.assertEqual(sink.flush_rows, 2)
        self.assertEqual(sink._buffer, [])
        self.assertIsNone(sink._writer)
        self.assertIsNone(sink._path)

    def test_append_and_flush_small(self):
        # set a small flush threshold for testing
        flush_rows = 2

        # create a DataSink with small flush threshold for testing
        sink = DataSink(out_dir=self.out_dir, owner_name=self.owner_name, data_name=self.small_data_name, flush_rows=flush_rows)

        # append small data and check that it flushes after threshold is reached
        for i, obs in enumerate(self.small_data):
            sink.append(obs)
            i_mod = i % flush_rows

            if (i + 1) % flush_rows == 0:
                # after every 2 appends, the data should be flushed and _rows should be cleared
                self.assertEqual(sink._buffer, [])
                self.assertIsNotNone(sink._writer)
                self.assertIsNotNone(sink._path)
            else:
                # before threshold is reached, _rows should contain the appended data
                self.assertEqual(sink._buffer[i_mod], self.small_data[i])
        
        # flush any remaining data and close the sink
        sink.close()
        
        # check that the parquet file was created
        self.assertTrue(os.path.exists(sink._path))

        # check that the sink is empty
        self.assertEqual(sink._buffer, [])
        self.assertIsNone(sink._writer)
        self.assertIsNotNone(sink._path)
        self.assertTrue(sink._closed)

        # load the parquet file and check that it contains the expected data
        loaded_data = pd.read_parquet(sink._path)

        # package the original small data into a DataFrame for comparison
        expected_data = pd.DataFrame(self.small_data)
        
        # the parquet file should contain all the data
        pd.testing.assert_frame_equal(loaded_data, expected_data)

    def test_append_and_flush_large(self):
        # set a small flush threshold for testing
        flush_rows = 100

        # create a DataSink with small flush threshold for testing
        sink = DataSink(out_dir=self.out_dir, owner_name=self.owner_name, data_name=self.large_data_name, flush_rows=flush_rows)

        # append large data and check that it flushes after threshold is reached
        for i, obs in enumerate(self.large_data):
            sink.append(obs)
            i_mod = i % flush_rows

            if (i + 1) % flush_rows == 0:
                # after every 100 appends, the data should be flushed and _rows should be cleared
                self.assertEqual(sink._buffer, [])
                self.assertIsNotNone(sink._writer)
                self.assertIsNotNone(sink._path)
            else:
                # before threshold is reached, _rows should contain the appended data
                self.assertEqual(sink._buffer[i_mod], self.large_data[i])
        
        # flush any remaining data
        sink.close()
        
        # check that the parquet file was created
        self.assertTrue(os.path.exists(sink._path))

        # check that the sink is empty
        self.assertEqual(sink._buffer, [])
        self.assertIsNone(sink._writer)
        self.assertIsNotNone(sink._path)
        self.assertTrue(sink._closed)

        # load the parquet file and check that it contains the expected data
        loaded_data = pd.read_parquet(sink._path)

        # package the original large data into a DataFrame for comparison
        expected_data = pd.DataFrame(self.large_data)
        
        # the parquet file should contain all the data
        pd.testing.assert_frame_equal(loaded_data, expected_data)

    def test_extend_and_flush_small(self):
        # set a small flush threshold for testing
        flush_rows = 2

        # create a DataSink with small flush threshold for testing
        sink = DataSink(out_dir=self.out_dir, owner_name=self.owner_name, data_name=self.small_data_name, flush_rows=flush_rows)

        # extend with all data at once
        sink.extend(list(self.small_data))  

        # check that only the last observation is in the buffer (since flush should have occurred after every 2 appends)
        self.assertEqual(len(sink._buffer), 1)
        self.assertEqual(sink._buffer, [self.small_data[-1]])
        self.assertIsNotNone(sink._writer)
        self.assertIsNotNone(sink._path)
        
        # flush any remaining data and close the sink
        sink.close()
        
        # check that the parquet file was created
        self.assertTrue(os.path.exists(sink._path))

        # check that the sink is empty
        self.assertEqual(sink._buffer, [])
        self.assertIsNone(sink._writer)
        self.assertIsNotNone(sink._path)
        self.assertTrue(sink._closed)

        # load the parquet file and check that it contains the expected data
        loaded_data = pd.read_parquet(sink._path)

        # package the original small data into a DataFrame for comparison
        expected_data = pd.DataFrame(self.small_data)
        
        # the parquet file should contain all the data
        pd.testing.assert_frame_equal(loaded_data, expected_data)

    def test_extend_and_flush_large(self):
        # set a small flush threshold for testing
        flush_rows = 900

        # create a DataSink with smaller flush threshold than the length of the data for testing
        sink = DataSink(out_dir=self.out_dir, owner_name=self.owner_name, data_name=self.large_data_name, flush_rows=flush_rows)

        # extend with all data at once
        sink.extend(list(self.large_data))  

        # check that only the last observation is in the buffer (since flush should have occurred after every 2 appends)
        self.assertEqual(len(sink._buffer), len(self.large_data) % flush_rows)
        self.assertEqual(sink._buffer, self.large_data[-(len(self.large_data) % flush_rows):])
        self.assertIsNotNone(sink._writer)
        self.assertIsNotNone(sink._path)
        
        # flush any remaining data and close the sink
        sink.close()
        
        # check that the parquet file was created
        self.assertTrue(os.path.exists(sink._path))

        # check that the sink is empty
        self.assertEqual(sink._buffer, [])
        self.assertIsNone(sink._writer)
        self.assertIsNotNone(sink._path)
        self.assertTrue(sink._closed)

        # load the parquet file and check that it contains the expected data
        loaded_data = pd.read_parquet(sink._path)

        # package the original large data into a DataFrame for comparison
        expected_data = pd.DataFrame(self.large_data)
        
        # the parquet file should contain all the data
        pd.testing.assert_frame_equal(loaded_data, expected_data)

    def test_del(self):
        # construct expected sink path
        sink_path = os.path.join(self.out_dir, f"{self.large_data_name}.parquet")

        # clear any existing file at the sink path
        if os.path.exists(sink_path):
            os.remove(sink_path)
        
        # set a large flush threshold to avoid automatic flushing during append
        flush_rows = 1000
        
        # define a function to create a DataSink, append data, and then drop the reference without closing
        def _create_and_drop():
            sink = DataSink(out_dir=self.out_dir,
                            owner_name=self.owner_name,
                            data_name=self.large_data_name,
                            flush_rows=flush_rows)
            for obs in self.large_data:
                sink.append(obs)
            # no close, no flush, rely on finalizer / __del__

        # call said function 
        _create_and_drop()

        # force garbage collection to trigger __del__
        gc.collect()  

        # check that the parquet file was created
        self.assertTrue(os.path.exists(sink_path))

        # load the parquet file and check that it contains the expected data
        loaded_data = pd.read_parquet(sink_path)

        # package the original large data into a DataFrame for comparison
        expected_data = pd.DataFrame(self.large_data)
        
        # the parquet file should contain all the data
        pd.testing.assert_frame_equal(loaded_data, expected_data)

if __name__ == '__main__':
    # print banner
    print_scenario_banner("`DataSink` Unit Tests")
    
    # run tests
    unittest.main()
    