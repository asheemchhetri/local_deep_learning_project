import torch
from transformers import BertTokenizer
from model_training import BertClaimClassifier  # Make sure this path is correct
import pandas as pd
from tabulate import tabulate

def predict_claims(model_path='article_model.pth', test_data_path='data/test_data.csv', num_samples=10):
    """
    :param model_path: The path to the pre-trained model file (.pth) to be loaded.
    :param test_data_path: The path to the CSV file containing the test data for making predictions.
    :param num_samples: The number of random samples to select from the test dataset for making predictions.
    :return: None. The function prints prediction results and accuracy on the sampled data.
    """
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertClaimClassifier(dropout=0.1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load tokenizer and test data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_data = pd.read_csv(test_data_path)

    # Select random samples
    samples = test_data.sample(n=num_samples, random_state=42)

    # Store results
    results = []
    status_map = {0: 'Approved', 1: 'Denied', 2: 'Pending'}
    severity_mapping = {'mild': 0, 'moderate': 1, 'severe': 2}

    # Make predictions
    for _, row in samples.iterrows():
        # Prepare input
        text = f"{row['diagnosis_description']} [SEP] {row['provider_notes']}"
        encoding = tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        severity = torch.tensor(
            severity_mapping[row['severity']],
            dtype=torch.long
        ).unsqueeze(0)

        risk_score = torch.tensor(row['risk_score'] / 100.0, dtype=torch.float).unsqueeze(0)

        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        severity = severity.to(device)
        risk_score = risk_score.to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, severity, risk_score)
            _, predicted = torch.max(outputs.data, 1)
            predicted_status = status_map[predicted.item()]

        # Store result
        results.append({
            'Diagnosis': row['diagnosis_description'][:50] + "...",
            'Severity': row['severity'],
            'Risk Score': row['risk_score'],
            'Actual Status': row['claim_status'],
            'Predicted Status': predicted_status,
            'Correct': "✓" if predicted_status == row['claim_status'] else "✗"
        })

    # Print results
    print("\nPrediction Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))

    # Calculate accuracy
    correct = sum(1 for r in results if r['Correct'] == "✓")
    print(f"\nAccuracy on samples: {100 * correct / len(results):.2f}%")

if __name__ == "__main__":
    predict_claims()
