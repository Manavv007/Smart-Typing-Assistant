# Smart Typing Assistant Using N-gram Language Models

## 1. Project Overview

The Smart Typing Assistant is an NLP project that predicts the next word a user is likely to type. For example, when the user enters **“I am going to”**, the system may suggest **“school,” “work,”** or **“college.”**

The project uses traditional statistical language modelling rather than a neural network. It learns patterns from a text dataset, calculates word-sequence probabilities, and returns the three most likely next-word suggestions.

The main goals are to:

- Build Unigram, Bigram, and Trigram language models.
- Use Trigram as the primary prediction model.
- Fall back to Bigram and Unigram when the available context is insufficient.
- Handle unseen sequences using smoothing and backoff.
- Build an interactive typing interface.
- Evaluate prediction quality with measurable NLP metrics.

## 2. Which N-gram Models Will We Use?

An N-gram is a sequence of *N* consecutive words.

### Unigram (N = 1)

A Unigram model considers individual words without considering their surrounding context. It can suggest generally frequent words, but it is weak at predicting a word that fits a particular phrase.

### Bigram (N = 2)

A Bigram model considers the previous one word. For example, after the word **“to”**, it learns which words commonly follow it.

### Trigram (N = 3)

A Trigram model considers the previous two words. For example, given the context **“going to”**, it learns which words commonly follow that pair.

### Recommended approach

Train all three models, but use **Trigram as the main predictor**, with Bigram and Unigram as fallback models. This makes it possible to compare model performance and handle cases where a longer context has not appeared in the training data.

## 3. How Does Training Work?

We will not train a neural network. A traditional N-gram language model learns by counting word sequences in a training corpus and converting those counts into probabilities.

Imagine a small training corpus containing these sentences:

- I am going to school.
- I am going to college.
- I am going to work.
- I am going to school.

When the context is **“going to”**, the next-word counts are:

| Candidate next word | Count | Probability |
|---|---:|---:|
| school | 2 | 2/4 = 0.50 |
| college | 1 | 1/4 = 0.25 |
| work | 1 | 1/4 = 0.25 |

The resulting suggestions would be:

1. school — 50%
2. college — 25%
3. work — 25%

These are illustrative counts from a tiny example, not results from the real dataset.

For a Trigram, the conditional probability is:

**P(w₃ | w₁, w₂) = C(w₁, w₂, w₃) / C(w₁, w₂)**

Here, `C` represents the number of times a sequence occurs in the training text.

### Smoothing and backoff

A sequence may be absent from the training corpus. A model that relies only on raw counts may then assign it a zero probability. We will use **Laplace smoothing** as a simple baseline and a **backoff strategy**:

1. Try Trigram predictions using the last two words.
2. If there is insufficient evidence, try Bigram predictions using the last word.
3. If that is also insufficient, fall back to Unigram frequency.

For a more advanced experiment, we can compare Laplace smoothing with interpolation or Kneser–Ney smoothing. The final method should be selected using validation results.

## 4. Which Dataset Will We Use?

### Primary dataset: WikiText-2

We recommend starting with **WikiText-2**, a dataset containing Wikipedia text. It is a manageable starting point for a student project and provides separate training, validation, and test splits.

Dataset page: https://huggingface.co/datasets/Salesforce/wikitext

Why use it?

- It provides real English text rather than a tiny hand-written sample.
- It is suitable for training statistical language models.
- Its predefined splits support a more reliable evaluation process.
- It is small enough to be a practical starting point for a prototype.

**Limitation:** Wikipedia text is often more formal than everyday messaging, so the suggestions may not feel exactly like those from a phone keyboard.

### Optional extension: conversational text

If time permits, we can compare WikiText-2 with an appropriately licensed conversational dataset. Conversational data may produce more natural suggestions for casual typing. Before using any additional dataset, check its licence, privacy considerations, and suitability for redistribution.

## 5. System Architecture

The project will follow this pipeline:

1. **Dataset:** Load the WikiText-2 training split.
2. **Preprocessing:** Split text into sentences, tokenize it, normalize text, and handle rare words.
3. **N-gram training engine:** Count Unigrams, Bigrams, and Trigrams.
4. **Probability calculation:** Calculate conditional probabilities and apply smoothing.
5. **Prediction engine:** Use Trigram first, then back off to Bigram or Unigram when needed.
6. **Suggestion ranking:** Return the top three candidate words.
7. **User interface:** Let users type, view suggestions, and select a suggested word.
8. **Evaluation:** Measure prediction quality, perplexity, model size, and response time.

## 6. Implementation Plan

### Step 1: Load and prepare the dataset

Load the training split. Split the text into sentences, tokenize words, and normalize capitalization consistently. Preserve sentence boundaries so the model does not learn artificial word sequences across unrelated sentences.

### Step 2: Build the vocabulary and N-gram counts

Count each individual word, every adjacent pair of words, and every consecutive sequence of three words. Consider mapping very rare words to an `<UNK>` token so the model can handle words outside its vocabulary.

### Step 3: Calculate probabilities

Calculate the conditional probability of a word given its context. Start with Laplace smoothing, then optionally experiment with more advanced smoothing methods.

### Step 4: Predict the next word

For a sentence such as **“I am going to”**, the model uses the last two words, **“going to,”** as the Trigram context. It retrieves candidate continuations, calculates their scores, and ranks them.

If the context is missing or too rare, the model backs off to Bigram and then Unigram predictions.

### Step 5: Return the top three suggestions

Display the three highest-ranked words. The user should be able to click a suggestion to append it to the text and continue typing.

### Step 6: Add optional autocorrection

Use edit distance to find plausible corrections for misspelled words, then rank correction candidates using N-gram probabilities. Keep autocorrection separate from next-word prediction because they solve different problems. This feature should offer suggestions rather than claim to guarantee grammatical corrections.

### Step 7: Build the interface

Use Streamlit to create a simple web application with:

- A text input area.
- Three next-word suggestions.
- Clickable suggestions that extend the input.
- An optional panel showing prediction scores.
- A model-comparison page with evaluation results.

## 7. Technology Stack

- **Programming language:** Python
- **NLP library:** NLTK, or custom Python code for the N-gram counting and probability calculations
- **Dataset:** WikiText-2
- **Interface:** Streamlit
- **Evaluation:** Python with metrics implemented or calculated using suitable libraries
- **Model storage:** Pickle or JSON for a small prototype, with care taken to load only trusted serialized files

For educational value, we should implement the central counting, probability, and backoff logic ourselves. NLTK can still be used for tokenization or as a reference implementation for comparison.

## 8. How Will We Evaluate the Project?

Reserve the test split for final evaluation. Use validation data to compare settings and choose the model configuration before testing.

| Experiment | Purpose |
|---|---|
| Unigram vs. Bigram vs. Trigram | Determine whether additional context improves predictions |
| With vs. without smoothing | Test how the model handles unseen sequences |
| Trigram only vs. backoff | Determine whether fallback models improve coverage |
| Top-1 accuracy | Measure how often the first suggestion is the actual next word |
| Top-3 accuracy | Measure how often the correct next word appears among the three suggestions |
| Perplexity | Measure how well the model assigns probabilities to unseen text |
| Prediction latency | Measure how quickly suggestions appear |
| Model size | Measure storage and memory requirements |

Accuracy must be measured against held-out text, not the same examples used to train the model. The preprocessing and evaluation rules should be applied consistently across all model variants.

## 9. Expected Deliverables

The final project should include:

1. A trained N-gram model.
2. A working next-word prediction engine.
3. A simple Streamlit interface.
4. Top-three word suggestions.
5. A backoff strategy and smoothing.
6. An evaluation report comparing Unigram, Bigram, and Trigram models.
7. Optional autocorrection.
8. A README explaining how to install, train, run, and evaluate the project.

## 10. Estimated Timeline

A basic prototype is achievable in approximately **1–2 days**, depending on experience and setup.

- **Phase 1:** Load and preprocess the dataset.
- **Phase 2:** Implement and test N-gram counts and probabilities.
- **Phase 3:** Add smoothing, backoff, and top-three predictions.
- **Phase 4:** Build the Streamlit interface.
- **Phase 5:** Evaluate the models and prepare results.
- **Phase 6 (optional):** Add autocorrection and advanced smoothing comparisons.

## 11. Final Recommendation

Build the **Smart Typing Assistant using Unigram, Bigram, and Trigram language models**, with Trigram as the primary model and Bigram/Unigram backoff for sparse contexts. Begin with WikiText-2, implement the statistical engine yourself, and use Streamlit for the interface.

The project will be stronger if it demonstrates not only that the suggestions work, but also **how well they work** through held-out accuracy, perplexity, latency, and comparisons between model variants. This provides a clear technical story for the project report and viva.
