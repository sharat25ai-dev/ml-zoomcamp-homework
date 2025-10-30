import pickle
import sklearn

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

data = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

lead = pipeline.predict_proba(data)[0, 1]
print(f'Probability of Successful Lead: {lead}')
if lead > 0.5:
    print("This lead is likely to be successful.")
else:
    print("This lead is unlikely to be successful.")