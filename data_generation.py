import pandas as pd
import numpy as np
from faker import Faker
import random
from tqdm import tqdm

# Initialize Faker and seeds for reproducibility
fake = Faker()
random.seed(42)
np.random.seed(42)

class MedicalConditionGenerator:
    """
    class MedicalConditionGenerator:
        """
    def __init__(self):
        self.body_systems = {
            'cardiovascular': ['heart', 'artery', 'vein', 'valve', 'aorta', 'ventricle', 'atrium'],
            'respiratory': ['lung', 'bronchi', 'trachea', 'pleura', 'alveoli', 'airway'],
            'digestive': ['stomach', 'intestine', 'colon', 'liver', 'gallbladder', 'pancreas'],
            'musculoskeletal': ['muscle', 'bone', 'joint', 'tendon', 'ligament', 'spine'],
            'nervous': ['brain', 'nerve', 'spinal cord', 'neuron', 'cerebral', 'neural'],
            'endocrine': ['thyroid', 'pancreas', 'adrenal', 'pituitary', 'hormone']
        }

        self.pathologies = {
            'inflammatory': ['inflammation', 'itis', 'swelling', 'edema'],
            'infectious': ['infection', 'abscess', 'sepsis'],
            'degenerative': ['degeneration', 'dystrophy', 'wear and tear'],
            'neoplastic': ['tumor', 'cancer', 'neoplasm'],
            'traumatic': ['injury', 'trauma', 'fracture'],
            'metabolic': ['disorder', 'deficiency', 'syndrome']
        }

        self.severity_modifiers = {
            'severe': ['severe', 'acute', 'critical', 'life-threatening'],
            'moderate': ['moderate', 'significant', 'persistent', 'chronic'],
            'mild': ['mild', 'minor', 'slight', 'early-stage']
        }

        self.onset_descriptors = [
            'sudden onset of', 'gradual development of',
            'intermittent episodes of', 'progressive'
        ]

        self.complications = {
            'severe': [
                'with multi-organ failure',
                'leading to septic shock',
                'complicated by cardiac arrest',
                'requiring mechanical ventilation'
            ],
            'moderate': [
                'with associated symptoms',
                'requiring hospitalization',
                'with moderate complications',
                'requiring close monitoring'
            ],
            'mild': [
                'without complications',
                'managed with outpatient care',
                'responding to medication',
                'with mild symptoms'
            ]
        }

    def generate_condition(self, severity):
        system = random.choice(list(self.body_systems.keys()))
        body_part = random.choice(self.body_systems[system])
        pathology = random.choice(list(self.pathologies.keys()))
        condition_term = random.choice(self.pathologies[pathology])
        severity_modifier = random.choice(self.severity_modifiers[severity])
        onset = random.choice(self.onset_descriptors)
        complication = random.choice(self.complications[severity])

        condition = f"{severity_modifier} {condition_term} of the {body_part} {complication}"

        return condition

def generate_provider_note(severity, condition):
    """
    :param severity: The severity level of the patient's condition. Expected values are 'severe', 'moderate', or 'mild'.
    :param condition: A description of the patient's condition.
    :return: A string containing a provider's note that includes the patient's vital signs, observed symptoms, and a medical assessment based on the provided severity and condition.
    """
    vitals_ranges = {
        'severe': {'bp': (150, 200), 'hr': (100, 140), 'temp': (38.5, 40.0)},
        'moderate': {'bp': (130, 150), 'hr': (80, 100), 'temp': (37.5, 38.5)},
        'mild': {'bp': (110, 130), 'hr': (60, 80), 'temp': (36.5, 37.5)}
    }

    vitals = vitals_ranges[severity]
    bp = f"{random.randint(*vitals['bp'])}/{random.randint(60, 90)}"
    hr = random.randint(*vitals['hr'])
    temp = round(random.uniform(*vitals['temp']), 1)

    symptoms = {
        'severe': ["experiencing severe pain", "in critical condition", "with acute distress"],
        'moderate': ["reporting moderate discomfort", "with noticeable symptoms", "requiring attention"],
        'mild': ["with mild symptoms", "experiencing slight discomfort", "reporting minor issues"]
    }

    assessments = {
        'severe': [
            "Immediate intervention required.", "Critical condition noted.", "High-risk patient."
        ],
        'moderate': [
            "Monitor and continue treatment.", "Patient stable but requires attention.", "Moderate risk."
        ],
        'mild': [
            "Condition stable.", "Low risk.", "Routine follow-up recommended."
        ]
    }

    notes = (
        f"Patient is {random.choice(symptoms[severity])} due to {condition}. "
        f"Vital signs: BP {bp}, HR {hr}, Temp {temp}°C. "
        f"Assessment: {random.choice(assessments[severity])}"
    )

    return notes

def assign_claim_status(severity, risk_score):
    """
    :param severity: A string indicating the severity of the claim. Expected values are 'severe', 'moderate', or 'mild'.
    :param risk_score: An integer representing the risk score associated with the claim. Higher scores indicate higher risk.
    :return: A string representing the status of the claim, which can be 'Approved', 'Denied', or 'Pending', based on the calculated probabilities.
    """
    # Base probabilities for claim status based on severity
    base_probs = {
        'severe': [0.8, 0.1, 0.1],   # Approved, Denied, Pending
        'moderate': [0.6, 0.2, 0.2],
        'mild': [0.4, 0.3, 0.3]
    }

    probs = base_probs[severity].copy()

    # Adjust probabilities based on risk score
    if risk_score > 80:
        probs[0] += 0.1  # Increase approval chance
        probs[1] -= 0.05  # Decrease denial chance
        probs[2] -= 0.05  # Decrease pending chance
    elif risk_score < 30:
        probs[0] -= 0.1  # Decrease approval chance
        probs[1] += 0.1  # Increase denial chance

    # Normalize probabilities
    total = sum(probs)
    probs = [p / total for p in probs]

    claim_status = np.random.choice(['Approved', 'Denied', 'Pending'], p=probs)

    return claim_status

def create_synthetic_dataset(n_rows=1000000):
    """
    :param n_rows: The number of synthetic medical claims records to generate. Default is 1,000,000.
    :return: A Pandas DataFrame containing the generated synthetic medical claims records. Each record includes fields such as claim_id, diagnosis_description, provider_notes, severity, risk_score, and claim_status.
    """
    print(f"Generating {n_rows} medical claims records...")

    condition_generator = MedicalConditionGenerator()
    data = []

    for _ in tqdm(range(n_rows)):
        severity = np.random.choice(['severe', 'moderate', 'mild'], p=[0.2, 0.5, 0.3])
        condition = condition_generator.generate_condition(severity)
        provider_note = generate_provider_note(severity, condition)

        # Risk score based on severity with some variation
        base_risk_scores = {'severe': 90, 'moderate': 60, 'mild': 30}
        risk_score = base_risk_scores[severity] + random.uniform(-10, 10)
        risk_score = max(0, min(100, risk_score))

        claim_status = assign_claim_status(severity, risk_score)

        record = {
            'claim_id': fake.uuid4(),
            'diagnosis_description': condition,
            'provider_notes': provider_note,
            'severity': severity,
            'risk_score': risk_score,
            'claim_status': claim_status
        }

        data.append(record)

    df = pd.DataFrame(data)
    print("\nData generation complete.")
    return df

if __name__ == "__main__":
    # Generate the dataset
    dataset_size = 1000000  # Adjust the size as needed
    df = create_synthetic_dataset(n_rows=dataset_size)

    # Display dataset statistics
    print("\nDataset Statistics:")
    print(f"Total Records: {len(df)}")
    print("\nClaim Status Distribution:")
    print(df['claim_status'].value_counts(normalize=True))

    print("\nSeverity Distribution:")
    print(df['severity'].value_counts(normalize=True))

    # Save the dataset to a CSV file
    df.to_csv('data/medical_claims_synthetic.csv', index=False)
    print("\nDataset saved as 'data/medical_claims_synthetic.csv'")
