import os
import sys
from typing import List
from pyspark.sql.types import TimestampType, StringType, StructType, StructField
from dataclasses import dataclass
from consumerComplaint.exception import ConsumerComplaintException


@dataclass
class FeatureProperties:
    col_name: str
    data_type: str

class FinanceDataSchema:

    def __init__(self):
        self.features = [
            FeatureProperties('company_response', 'string'),
            FeatureProperties('consumer_consent_provided', 'string'),
            FeatureProperties('submitted_via', 'string'),
            FeatureProperties('timely', 'string'),
            FeatureProperties('date_sent_to_company', 'timestamp'),
            FeatureProperties('date_received', 'timestamp'),
            FeatureProperties('company', 'string'),
            FeatureProperties('issue', 'string'),
            FeatureProperties('product', 'string'),
            FeatureProperties('state', 'string'),
            FeatureProperties('zip_code', 'string'),
            FeatureProperties('consumer_disputed', 'string')
        ]

    @property
    def dataframe_schema(self) -> StructType:
        try:
            schema = StructType([
                StructField(feature.col_name, TimestampType() if feature.data_type == 'timestamp' else StringType())
                for feature in self.features
            ])
            return schema
        except Exception as e:
            raise ConsumerComplaintException(e, sys) from e
        
    @property
    def target_column(self) -> str:
        return 'consumer_disputed'

    @property
    def one_hot_encoding_features(self) -> List[str]:
        return ['company_response', 
                'consumer_consent_provided', 
                'submitted_via']

    @property
    def im_one_hot_encoding_features(self) -> List[str]:
        return [f"im_{col}" for col in self.one_hot_encoding_features]

    @property
    def string_indexer_one_hot_features(self) -> List[str]:
        return [f"si_{col}" for col in self.one_hot_encoding_features]

    @property
    def tf_one_hot_encoding_features(self) -> List[str]:
        return [f"tf_{col}" for col in self.one_hot_encoding_features]

    @property
    def tfidf_features(self) -> List[str]:
        return ["issue"]

    @property
    def derived_input_features(self) -> List[str]:
        features = [    
            "date_sent_to_company",
             "date_received"
        ]
        return features

    @property
    def derived_output_features(self) -> List[str]:
        return ["diff_in_days"]

    @property
    def numerical_columns(self) -> List[str]:
        return self.derived_output_features

    @property
    def im_numerical_columns(self) -> List[str]:
        return [f"im_{col}" for col in self.numerical_columns]

    @property
    def tfidf_feature(self) -> List[str]:
        return ["issue"]

    @property
    def tf_tfidf_features(self) -> List[str]:
        return [f"tf_{col}" for col in self.tfidf_feature]

    @property
    def input_features(self) -> List[str]:
        in_features = self.tf_one_hot_encoding_features + self.im_numerical_columns + self.tf_tfidf_features
        return in_features

    @property
    def required_columns(self) -> List[str]:
        features = [self.target_column] + self.one_hot_encoding_features + self.tfidf_features + \
                   ["date_sent_to_company", "date_received"]
        return features

    @property
    def required_prediction_columns(self) -> List[str]:
        features =  self.one_hot_encoding_features + self.tfidf_features + \
                   ["date_sent_to_company", "date_received"]
        return features



    @property
    def unwanted_columns(self) -> List[str]:
        features = ["complaint_id",
                    "sub_product",  
                    "complaint_what_happened"]

        return features

    @property
    def vector_assembler_output(self) -> str:
        return "va_input_features"

    @property
    def scaled_vector_input_features(self) -> str:
        return "scaled_input_features"

    @property
    def target_indexed_label(self) -> str:
        return f"indexed_{[self.target_column]}"

    @property
    def prediction_column_name(self) -> str:
        return "prediction"

    @property
    def prediction_label_column_name(self) -> str:
        return f"{self.prediction_column_name}_{[self.target_column]}"
    
    @property
    def get_input_columns(self) -> str :
        return [
            "company_response",
            "consumer_consent_provided",
            "state",
            "sub_issue",
            "zip_code"
        ]
    
    @property
    def get_output_columns(self) ->str:
        return [
            f"im_{col}" for col in self.get_input_columns
        ]

