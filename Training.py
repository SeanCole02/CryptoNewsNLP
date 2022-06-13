import os
import random
import spacy
from spacy.util import minibatch, compounding, decaying
from spacy.pipeline.textcat import single_label_cnn_config, DEFAULT_SINGLE_TEXTCAT_MODEL
from spacy.training import Example

with open('news.txt', encoding='utf-8') as TEST_REVIEW:
    TEST_REVIEW = TEST_REVIEW.read()
def load_training_data(
    data_directory: str = "crypto/train",
    split: float = 0.8,
    limit: int = 0
) -> tuple:
    # Load from files
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding='utf-8') as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label}
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]
def evaluate_model(
    tokenizer, textcat, test_data: list
) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives, true_negatives = 0, 0
    false_positives, false_negatives = 1e-8, 1e-8  # Can't be 0 because of presence in denominator
    #true_negatives = 0
    #false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if (
                predicted_label == "neg"
            ):
                continue
            if score >= 0.5 and true_label['cats']["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label['cats']["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label['cats']["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label['cats']["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    print(true_positives, false_positives, true_negatives, false_negatives)
    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 100
) -> None:
    # Build pipeline
    examples = []
    examplesa = examples.append

    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe('textcat',last=True)
    textcat.add_label("pos")
    textcat.add_label("neg")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    #with nlp.disable_pipes(training_excluded_pipes):
    with nlp.select_pipes(enable="textcat"):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score")
        batch_sizes = compounding(
            1, 32, 1.005 #default 4, 32, 1.001 , 1.001 is how fast it compounds to the max
        )  # A generator that yields infinite series of input numbers
        for i in range(iterations):
            print(f"Training iteration {i}")
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                doc = nlp.make_doc(text[0])
                example = Example.from_dict(doc, labels[0])
                examplesa(example)
                nlp.update([example], drop=0.15, sgd=optimizer, losses=loss) #default drop 0.2 (20% neurons are dropped)
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    # Save model
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")

def test_model(input_data: str = TEST_REVIEW):
    #  Load saved trained model

    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        #f"Predicted sentiment: {prediction}"
        f"\tProbability: {float(score)*100}%"
    )

if __name__ == "__main__":
    train, test = load_training_data(limit=5000) #default 2500
    train_model(train, test) #This is to retrain the model, dont do it if you dont have to
    test_model()