"""Project definitions; mappings follow specification section 6 and UCI metadata."""
PROJECT_CONFIG = {"target_column":"Diabetes_binary","random_state":42,"test_size":0.20,"confidence_level":0.95,"primary_ml_dataset_variant":"unique_profile_v1","descriptive_dataset_variant":"full_clean_v1"}
MODEL_CONFIG = {"models":["logistic_regression","random_forest","gradient_boosting"],"default_threshold":0.50}
ANALYSIS_CONFIG = {"multiple_comparison_method":"fdr_bh","correlation_method":"spearman","calibration_bins":10}
TARGET_DEFINITION = "0 = neither prediabetes nor diabetes; 1 = prediabetes or diabetes (project-defined outcome)"
FEATURE_COLUMNS = ['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']
EXPECTED_COLUMNS = FEATURE_COLUMNS + ['Diabetes_binary']
REFERENCE_DATASET = {'rows':253680,'columns':22,'missing_cells':0,'duplicate_rows':24206,'target_counts':{0:218334,1:35346}}

def _variable(label, description, kind, labels=None, prediction_input=False, unit=None, minimum=None, maximum=None):
    return {'label':label,'description':description,'type':kind,'role':'feature','allowed_values':list(labels) if labels else None,'category_labels':labels,'unit':unit,'prediction_input':prediction_input,'required_for_model':True,'required_for_statistics':True,'visualization_type':'bar' if labels else 'histogram','minimum':minimum,'maximum':maximum,'integer_only':kind != 'continuous'}

_BINARY_DESCRIPTIONS = {
 'HighBP':('High blood pressure','Reported high blood pressure.'),
 'HighChol':('High cholesterol','Reported high cholesterol.'),
 'CholCheck':('Cholesterol check','Cholesterol checked within five years.'),
 'Smoker':('Smoking history','At least 100 cigarettes smoked in lifetime; not necessarily current smoking.'),
 'Stroke':('Stroke history','Ever told respondent had a stroke.'),
 'HeartDiseaseorAttack':('Heart disease or attack','Coronary heart disease or myocardial infarction.'),
 'PhysActivity':('Physical activity','Activity in past 30 days, excluding work.'),
 'Fruits':('Daily fruit consumption','Fruit consumed one or more times per day.'),
 'Veggies':('Daily vegetable consumption','Vegetables consumed one or more times per day.'),
 'HvyAlcoholConsump':('Heavy alcohol consumption','More than 14 drinks/week for adult men or 7 for adult women.'),
 'AnyHealthcare':('Healthcare coverage','Any healthcare coverage including insurance or prepaid plans.'),
 'NoDocbcCost':('Cost prevented doctor visit','Needed a doctor in past 12 months but could not attend due to cost.'),
 'DiffWalk':('Difficulty walking','Serious difficulty walking or climbing stairs.'),
 'Sex':('Sex','Source survey sex category.'),
}
VARIABLE_METADATA = {c:_variable(label,desc,'binary',{0:'No',1:'Yes'},c in {'HighBP','HighChol','Smoker','PhysActivity','Sex'}) for c,(label,desc) in _BINARY_DESCRIPTIONS.items()}
VARIABLE_METADATA['Sex']['category_labels'] = {0:'Female',1:'Male'}
VARIABLE_METADATA['Diabetes_binary'] = _variable('Prediabetes-or-diabetes outcome',TARGET_DEFINITION,'binary',{0:'Neither prediabetes nor diabetes',1:'Prediabetes or diabetes'})
VARIABLE_METADATA['Diabetes_binary']['role'] = 'target'
VARIABLE_METADATA.update({
 'BMI':_variable('Body mass index','Recorded BMI; observed range 12–98 is not a clinical validity threshold.','continuous',prediction_input=True,unit='BMI units',minimum=0),
 'MentHlth':_variable('Poor mental-health days','Poor mental-health days in past 30 days; zero is valid.','discrete_numeric',unit='days',minimum=0,maximum=30),
 'PhysHlth':_variable('Poor physical-health days','Poor physical-health days in past 30 days; zero is valid.','discrete_numeric',unit='days',minimum=0,maximum=30),
 'GenHlth':_variable('General health','Ordered self-reported general-health rating.','ordinal',{1:'Excellent',2:'Very good',3:'Good',4:'Fair',5:'Poor'},True),
 'Age':_variable('Age group','Grouped age, not age in years.','ordinal',dict(enumerate(['18–24','25–29','30–34','35–39','40–44','45–49','50–54','55–59','60–64','65–69','70–74','75–79','80+'],1)),True),
 'Education':_variable('Education','Highest completed education category.','ordinal',{1:'No schooling / kindergarten only',2:'Grades 1–8',3:'Grades 9–11',4:'Grade 12 / GED',5:'Some college / technical school',6:'College graduate / 4+ years'}),
 'Income':_variable('Household income','Annual household income in USD; brackets, not exact amounts.','ordinal',{1:'Under USD 10,000',2:'USD 10,000–<15,000',3:'USD 15,000–<20,000',4:'USD 20,000–<25,000',5:'USD 25,000–<35,000',6:'USD 35,000–<50,000',7:'USD 50,000–<75,000',8:'USD 75,000+'})
})
