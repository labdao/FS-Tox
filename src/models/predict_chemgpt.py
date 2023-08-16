# Set performance outcomes
baseline_auc_pr = []
finetuned_auc_pr = []
baseline_accuracy = []
finetuned_accuracy = []

# Load baseline and finetuned models 
finetuned_model = AutoModelForSequenceClassification.from_pretrained("./results", num_labels=2)
finetuned_model.config.pad_token_id = tokenizer.pad_token_id

baseline_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
baseline_model.config.pad_token_id = tokenizer.pad_token_id

# Generate predictions
base_acc, base_auc = generate_predictions(baseline_model, query_dataset)
finetune_acc, finetune_auc = generate_predictions(finetuned_model, query_dataset)
baseline_accuracy.append(base_acc)
finetuned_accuracy.append(finetune_acc)
baseline_auc_pr.append(base_auc)
finetuned_auc_pr.append(finetune_auc)


def generate_predictions(model, query_dataset):

    # Define test trainer
    trainer = Trainer(model)

    # Make prediction
    raw_pred, _, _ = trainer.predict(query_dataset)
    
    # Assuming true_labels is a numpy array containing the ground truth labels for query_dataset
    true_labels = np.array(query_dataset.labels)

    # Compute the probabilities for the positive class
    y_prob = raw_pred[:, 1]
    
    y_pred = np.argmax(raw_pred, axis=1)

    # Compute AUC-PR
    return accuracy_score(true_labels, y_pred), average_precision_score(true_labels, y_prob)