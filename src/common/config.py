PROJECT_CONFIG = {"target_column":"Diabetes_binary","random_state":42,"test_size":0.20,"confidence_level":0.95}
MODEL_CONFIG = {"models":["logistic_regression","random_forest","gradient_boosting"],"default_threshold":0.50}
ANALYSIS_CONFIG = {"multiple_comparison_method":"fdr_bh","correlation_method":"spearman","calibration_bins":10}

# Verify exact coding against the supplied dataset before implementation.
_BINARY = lambda label, pred=False: {"label":label,"type":"binary","role":"feature","allowed_values":[0,1],"category_labels":{0:"No",1:"Yes"},"unit":None,"prediction_input":pred,"required_for_model":True,"required_for_statistics":True,"visualization_type":"bar"}
VARIABLE_METADATA = {
    "Diabetes_binary":{"label":"Diabetes Status","description":"Binary diabetes outcome.","type":"binary","role":"target","allowed_values":[0,1],"category_labels":{0:"No diabetes",1:"Diabetes"},"unit":None,"prediction_input":False,"required_for_model":True,"required_for_statistics":True,"visualization_type":"bar"},
    "HighBP":_BINARY("High Blood Pressure",True), "HighChol":_BINARY("High Cholesterol",True),
    "CholCheck":_BINARY("Cholesterol Check"), "Smoker":_BINARY("Smoking History",True),
    "Stroke":_BINARY("History of Stroke"), "HeartDiseaseorAttack":_BINARY("Heart Disease or Heart Attack"),
    "PhysActivity":_BINARY("Physical Activity",True), "Fruits":_BINARY("Fruit Consumption"),
    "Veggies":_BINARY("Vegetable Consumption"), "HvyAlcoholConsump":_BINARY("Heavy Alcohol Consumption"),
    "AnyHealthcare":_BINARY("Healthcare Coverage"), "NoDocbcCost":_BINARY("Could Not See Doctor Due to Cost"),
    "DiffWalk":_BINARY("Difficulty Walking"), "Sex":_BINARY("Sex",True),
    "BMI":{"label":"Body Mass Index","description":"Body mass index value.","type":"continuous","role":"feature","allowed_values":None,"category_labels":None,"unit":"BMI units","prediction_input":True,"required_for_model":True,"required_for_statistics":True,"visualization_type":"histogram"},
    "GenHlth":{"label":"General Health","description":"Self-reported general health category.","type":"ordinal","role":"feature","allowed_values":[1,2,3,4,5],"category_labels":{1:"Excellent",2:"Very good",3:"Good",4:"Fair",5:"Poor"},"unit":None,"prediction_input":True,"required_for_model":True,"required_for_statistics":True,"visualization_type":"bar"},
    "MentHlth":{"label":"Poor Mental Health Days","description":"Number of poor mental-health days.","type":"discrete_numeric","role":"feature","allowed_values":None,"category_labels":None,"unit":"days","prediction_input":False,"required_for_model":True,"required_for_statistics":True,"visualization_type":"histogram"},
    "PhysHlth":{"label":"Poor Physical Health Days","description":"Number of poor physical-health days.","type":"discrete_numeric","role":"feature","allowed_values":None,"category_labels":None,"unit":"days","prediction_input":False,"required_for_model":True,"required_for_statistics":True,"visualization_type":"histogram"},
    "Age":{"label":"Age Group","description":"Numerically encoded age category; verify exact labels.","type":"ordinal","role":"feature","allowed_values":list(range(1,14)),"category_labels":None,"unit":None,"prediction_input":True,"required_for_model":True,"required_for_statistics":True,"visualization_type":"bar"},
    "Education":{"label":"Education Level","description":"Ordinal education category; verify exact labels.","type":"ordinal","role":"feature","allowed_values":[1,2,3,4,5,6],"category_labels":None,"unit":None,"prediction_input":False,"required_for_model":True,"required_for_statistics":True,"visualization_type":"bar"},
    "Income":{"label":"Income Category","description":"Ordinal income category; verify exact labels.","type":"ordinal","role":"feature","allowed_values":list(range(1,9)),"category_labels":None,"unit":None,"prediction_input":False,"required_for_model":True,"required_for_statistics":True,"visualization_type":"bar"},
}
