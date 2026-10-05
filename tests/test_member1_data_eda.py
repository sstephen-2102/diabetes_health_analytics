"""Meaningful data/EDA contract tests, runnable with unittest or pytest."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.common.config import EXPECTED_COLUMNS, FEATURE_COLUMNS, VARIABLE_METADATA
from src.common.exceptions import DataLoadError, DataValidationError, InvalidParameterError, MissingColumnError
from src.data.loader import load_dataset, download_kaggle_dataset
from src.data.validation import validate_dataset, validate_variable
from src.data.quality import generate_data_quality_report
from src.data.preprocessing import prepare_analysis_data
from src.statistics.descriptive import summarize_numeric_variable, summarize_categorical_variable, compare_by_target, calculate_prevalence_table
from src.statistics.association import calculate_correlation_matrix
from src.analysis.eda import compare_dataset_variants
from app.services import get_dashboard_summary, get_eda_summary


def fixture():
    # Repeated complete profiles, and two equal selected features with distinct
    # original profiles, detect deduplication at the wrong stage.
    data = pd.DataFrame({c:[0,0,0,0] for c in EXPECTED_COLUMNS})
    for c in ['GenHlth','Age','Education','Income']:
        data[c] = [1,1,2,3]
    data['BMI'] = [20,20,30,30]
    data['Diabetes_binary'] = [0,0,1,1]
    data['MentHlth'] = [0,0,2,30]
    data['PhysHlth'] = [0,0,4,30]
    data['HighBP'] = [0,0,1,1]
    return data


class LoaderTests(unittest.TestCase):
    def test_load_keeps_rows_and_records_source(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'sample.csv'
            fixture().to_csv(path,index=False)
            data = load_dataset(str(path))
            assert_frame_equal(data,fixture())
            self.assertEqual(data.attrs['dataset_variant'],'raw_v1')
            self.assertEqual(data.attrs['source_file'],'sample.csv')

    def test_missing_empty_malformed_duplicate_header(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'sample.csv'
            with self.assertRaises(DataLoadError): load_dataset(str(path))
            for content in ['', 'a,b\n', 'a,b\n1,2\n3,4,5\n', 'a,b\ntext,1\n', 'a,a\n1,2\n']:
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(DataLoadError):
                    load_dataset(str(path))

    def test_kaggle_copies_nested_csv_and_refuses_raw_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            source,dest = Path(temp)/'source',Path(temp)/'dest'
            (source/'nested').mkdir(parents=True)
            (source/'nested'/'sample.csv').write_text('a,b\n1,2\n')
            fake = types.SimpleNamespace(dataset_download=lambda _:str(source))
            with patch.dict(sys.modules,{'kagglehub':fake}):
                download_kaggle_dataset('owner/data',str(dest))
                download_kaggle_dataset('owner/data',str(dest))
                (dest/'nested'/'sample.csv').write_text('a,b\n9,9\n')
                with self.assertRaises(DataLoadError): download_kaggle_dataset('owner/data',str(dest))

    def test_kaggle_invalid_id(self):
        with self.assertRaises(DataLoadError): download_kaggle_dataset('bad','unused')


class ValidationTests(unittest.TestCase):
    def test_valid_and_reference_drift_separate(self):
        result = validate_dataset(fixture(),EXPECTED_COLUMNS,'Diabetes_binary')
        self.assertTrue(result['is_valid'])
        self.assertFalse(result['matches_reference'])
        self.assertEqual(result['duplicate_rows'],1)

    def test_missing_extra_columns_and_empty(self):
        for frame in [fixture().drop(columns='Age'), fixture().assign(extra=1), fixture().iloc[:0]]:
            self.assertFalse(validate_dataset(frame,EXPECTED_COLUMNS,'Diabetes_binary')['is_valid'])

    def test_invalid_category_fractional_days_infinity_missing_text(self):
        for column,value in [('Age',14),('Sex',2),('MentHlth',31),('PhysHlth',.5),('BMI',0),('BMI',np.inf),('Income',np.nan)]:
            frame = fixture().astype({column:float})
            frame.loc[0,column] = value
            self.assertFalse(validate_variable(frame,column,VARIABLE_METADATA)['is_valid'])
        frame = fixture().astype({'Sex':str})
        self.assertFalse(validate_variable(frame,'Sex',VARIABLE_METADATA)['is_valid'])

    def test_valid_endpoints_and_bmi_extreme_retained(self):
        frame = fixture()
        frame.loc[0,'BMI'] = 98
        for col in ['BMI','MentHlth','PhysHlth']:
            self.assertTrue(validate_variable(frame,col,VARIABLE_METADATA)['is_valid'])

    def test_single_target_class_warns(self):
        frame = fixture().assign(Diabetes_binary=0)
        report = validate_dataset(frame,EXPECTED_COLUMNS,'Diabetes_binary')
        self.assertTrue(report['is_valid'])
        self.assertTrue(any('Single-class' in w for w in report['warnings']))

    def test_duplicate_column_names_raise(self):
        frame = pd.DataFrame([[1,2]],columns=['a','a'])
        with self.assertRaises(DataValidationError): validate_dataset(frame,['a'],'a')


class VariantTests(unittest.TestCase):
    def test_variants_do_not_mutate_source(self):
        frame,original = fixture(),fixture()
        full = prepare_analysis_data(frame,'Diabetes_binary',FEATURE_COLUMNS)
        unique = prepare_analysis_data(frame,'Diabetes_binary',FEATURE_COLUMNS,dataset_variant='unique_profile_v1')
        self.assertEqual(len(full),4)
        self.assertEqual(len(unique),3)
        self.assertEqual(unique.attrs['dropped_duplicate_rows'],1)
        full.loc[0,'BMI'] = 999
        assert_frame_equal(frame,original)

    def test_complete_row_dedup_before_feature_selection(self):
        selected = prepare_analysis_data(fixture(),'Diabetes_binary',['BMI'],dataset_variant='unique_profile_v1')
        self.assertEqual(len(selected),3)
        self.assertEqual(selected.duplicated().sum(),1)

    def test_missing_handling_explicit(self):
        frame = fixture().astype({'BMI':float})
        frame.loc[0,'BMI'] = np.nan
        self.assertEqual(len(prepare_analysis_data(frame,'Diabetes_binary',['BMI'])),4)
        result = prepare_analysis_data(frame,'Diabetes_binary',['BMI'],drop_missing=True)
        self.assertEqual(len(result),3)
        self.assertEqual(result.attrs['dropped_missing_rows'],1)

    def test_invalid_variant_feature_target_and_missing_target(self):
        with self.assertRaises(InvalidParameterError): prepare_analysis_data(fixture(),'Diabetes_binary',['BMI'],dataset_variant='balanced_train_v1')
        with self.assertRaises(InvalidParameterError): prepare_analysis_data(fixture(),'Diabetes_binary',['Diabetes_binary'])
        with self.assertRaises(MissingColumnError): prepare_analysis_data(fixture(),'Diabetes_binary',['absent'])
        with self.assertRaises(DataValidationError): prepare_analysis_data(fixture().assign(Diabetes_binary=2),'Diabetes_binary',['BMI'])


class SummaryTests(unittest.TestCase):
    def test_numeric_known_values(self):
        frame = pd.DataFrame({'BMI':[1.,2.,3.,4.]})
        result = summarize_numeric_variable(frame,'BMI')
        self.assertEqual(result['mean'],2.5)
        self.assertEqual(result['median'],2.5)
        self.assertAlmostEqual(result['variance'],5/3)
        self.assertEqual(result['iqr'],1.5)

    def test_numeric_singleton_all_missing_and_invalid(self):
        self.assertIsNone(summarize_numeric_variable(pd.DataFrame({'BMI':[2.]}),'BMI')['std'])
        self.assertIsNone(summarize_numeric_variable(pd.DataFrame({'BMI':[np.nan]}),'BMI')['mean'])
        with self.assertRaises(InvalidParameterError): summarize_numeric_variable(fixture(),'Age')
        with self.assertRaises(DataValidationError): summarize_numeric_variable(pd.DataFrame({'BMI':[np.inf]}),'BMI')

    def test_category_missing_denominator_and_absent_levels(self):
        frame = pd.DataFrame({'Sex':[0.,0.,np.nan]})
        table = summarize_categorical_variable(frame,'Sex')
        self.assertEqual(table['count'].sum(),3)
        self.assertAlmostEqual(table['percentage'].sum(),100)
        self.assertEqual(table.loc[table.category==1,'count'].item(),0)
        self.assertNotIn('percentage',summarize_categorical_variable(frame,'Sex',False))

    def test_prevalence_uses_within_feature_denominator(self):
        frame = pd.DataFrame({'HighBP':[0,0,0,1],'Diabetes_binary':[0,0,1,1]})
        table = calculate_prevalence_table(frame,'HighBP','Diabetes_binary').set_index('category')
        self.assertAlmostEqual(table.loc[0,'target_percentage'],100/3)
        self.assertEqual(table.loc[1,'target_percentage'],100)

    def test_absent_category_is_undefined_not_zero_prevalence(self):
        table = calculate_prevalence_table(fixture(),'Sex','Diabetes_binary').set_index('category')
        self.assertTrue(pd.isna(table.loc[1,'target_percentage']))
        self.assertEqual(table.loc[1,'total_observations'],0)

    def test_missing_category_not_hidden(self):
        frame = fixture().astype({'Sex':float})
        frame.loc[0,'Sex'] = np.nan
        table = calculate_prevalence_table(frame,'Sex','Diabetes_binary')
        self.assertEqual(table.total_observations.sum(),4)
        self.assertIn('Missing',table.label.values)

    def test_comparison_handles_ordinal_as_categories(self):
        table = compare_by_target(fixture(),'Age','Diabetes_binary')
        self.assertIn('category',table)
        self.assertNotIn('mean',table)
        self.assertEqual(table['count'].sum(),4)

    def test_target_validation_and_target_feature_rejected(self):
        with self.assertRaises(DataValidationError): calculate_prevalence_table(fixture().assign(Diabetes_binary=2),'Sex','Diabetes_binary')
        with self.assertRaises(InvalidParameterError): compare_by_target(fixture(),'Diabetes_binary','Diabetes_binary')

    def test_quality_repeated_rows_and_only_bmi_outliers(self):
        report = generate_data_quality_report(fixture())
        self.assertEqual(report['duplicate_rows'],1)
        self.assertEqual(report['rows_in_repeated_patterns'],2)
        self.assertEqual(report['distinct_patterns'],3)
        self.assertNotIn('MentHlth',report['potential_outliers'])
        self.assertEqual(generate_data_quality_report(fixture().iloc[:0])['duplicate_rate_percent'],0)

    def test_spearman_ties_and_constants(self):
        frame = pd.DataFrame({'a':[1,1,2,3],'b':[3,3,2,1],'constant':[1,1,1,1]})
        matrix = calculate_correlation_matrix(frame,list(frame))
        self.assertAlmostEqual(matrix.loc['a','b'],-1)
        self.assertTrue(pd.isna(matrix.loc['a','constant']))
        with self.assertRaises(InvalidParameterError): calculate_correlation_matrix(frame,['a'],'unknown')

    def test_variant_sensitivity_and_services(self):
        result = compare_dataset_variants(fixture())
        overall = result['sensitivity'].query("feature == 'Diabetes_binary'").iloc[0]
        self.assertAlmostEqual(overall.estimate_full,50)
        self.assertAlmostEqual(overall.estimate_unique,200/3)
        self.assertAlmostEqual(overall.change_unique_minus_full,50/3)
        self.assertEqual(get_dashboard_summary(fixture(),'Diabetes_binary')['status'],'success')
        self.assertIn('prevalence',get_eda_summary(fixture(),'Age'))
        self.assertNotIn('prevalence',get_eda_summary(fixture(),'BMI'))


if __name__ == '__main__':
    unittest.main()
