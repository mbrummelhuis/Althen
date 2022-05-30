import os
import pandas as pd
from init import secrets as s
from blockbax_sdk import errors, HttpClient, models
from dataclasses import asdict
from datetime import datetime, timedelta
from itertools import chain

class BlockbaxConnector(HttpClient):
    def __init__(self, access_token, project_id):
        try:
            self.client = HttpClient(access_token, project_id)
        except errors.BlockbaxHTTPError:
            print('Blockbax HTTP Response Error with the following data:')
            raise
        except Exception:
            raise

    def get_bricks(self, filter=None):
        # Retrieve all brick data
        all_bricks = self.client.get_subjects()
        subject_types = self.client.get_subject_types()
        property_types = self.client.get_property_types()
        
        # convert list of subjects to dataframe
        bricks = pd.DataFrame([asdict(brick) for brick in all_bricks])

        # merge with subject type names
        sub_types = pd.DataFrame([{'subject_type_name': type.name, 'subject_type_id': type.id} for type in subject_types])
        bricks = bricks.merge(sub_types, how='left', on='subject_type_id')

        features = ['name', 'id', 'created_date', 'updated_date', 'subject_type_name', 'subject_type_id']
        return bricks[features]

    def get_metrics(self, filter={'name': None, 'external_id': None, 'subject_type_ids': None}):
        # retrieve all metrics
        all_metrics = self.client.get_metrics(filter['name'], filter['external_id'], filter['subject_type_ids'])
        # convert features to dataframe
        return pd.DataFrame([{'subject_type_id': metric.subject_type_id, 'metric_id': metric.id, 'metric_name': metric.external_id} for metric in all_metrics])

    def get_measurements(self, id, metric):
            series = self.client.get_measurements(subject_ids=[id], metric_ids=[metric], from_date=datetime.utcnow() - timedelta(days=365))
            if not series:
                print(f"Retrieved empty list for subject {id} and metric {metric}.")
                return {'subject_id': id, 'metric_id': metric, 'measurements': None}
            elif len(series) == 1:
                s = series[0]
                print(f"Retrieved {len(s.measurements)} measurements for subject {s.subject_id} and metric {s.metric_id}.")
                return asdict(s)
            else:
                raise Exception(f'Expected one series returned but received {len(series)}.')
    
    def get_all_data(self):
        try:
            # retrieve all bricks
            df_bricks = self.get_bricks()
            # retrieve metrics for bricks
            df_metrics = self.get_metrics()
            # combine bricks and metrics
            df_combined = df_bricks.merge(df_metrics, how='left', on='subject_type_id').drop(columns=['subject_type_id'])
            # iterating over two columns, use `zip`
            df_measurements = pd.DataFrame([self.get_measurements(id, metric) for id, metric in zip(df_combined['id'], df_combined['metric_id'])])
            # merge with measurements
            df_combined = df_combined.merge(df_measurements, left_on=['id', 'metric_id'], right_on=['subject_id', 'metric_id']).drop(columns=['metric_id', 'subject_id'])

            # explode measurements to rows
            df_combined = df_combined.explode('measurements').reset_index(drop=True)
            # split dict column into seperate columns
            df_combined = df_combined.join(pd.json_normalize(df_combined['measurements'])).drop(columns=['measurements'])
            # unstack metrics into columns
            data = pd.pivot_table(df_combined[['id', 'metric_name', 'date', 'number']],\
                index=['id', 'date'], columns='metric_name', values='number', aggfunc='first')         
            return data
        except errors.BlockbaxHTTPError as ex:
            print('Blockbax HTTP Response Error with the following data:')
            raise
        except Exception:
            raise
        
if __name__ == '__main__':
    client = BlockbaxConnector(s.access_token, s.project_id)
    bricks = client.get_bricks()
    data = client.get_all_data()

    # save to csv
    dir = os.path.dirname(__file__)
    bricks.to_csv(os.path.join(dir, "bricks_df.csv"))
    data.to_csv(os.path.join(dir, "measurements_df.csv"))
    print(data.head(5))