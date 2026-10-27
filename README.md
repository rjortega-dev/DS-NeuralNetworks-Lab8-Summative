# EcoSort Waste Management Assistant

An integrated waste management assistant built as a summative lab for a Data Science and AI bootcamp. Given either a **photo** or a **written description** of a waste item, it identifies the waste category and generates recycling instructions grounded in (fictional) city policy documents.

Everything lives in one notebook: `waste_management_summative.ipynb` (outputs included, so it can be read without re-running).

## How it works

```
image ──► CNN (MobileNetV2, frozen base) ────┐
                                             ├─► (category, confidence) ──► confidence gate (≥ 0.5)
text ───► TF-IDF + logistic regression ──────┘                                      │
                                                                                    ▼
          category ──► retriever (MiniLM embeddings, category-filtered) ──► fine-tuned GPT-2
                                                                                    │
                                                      instructions + retrieved policy documents
```

Both classifiers share the same 9 labels and class order, so the generator doesn't care which one produced the category. The entry point, `waste_management_assistant(input_data, input_type)`, never raises on bad input; it returns an explanatory message instead.

## Results

| Component | Approach | Result |
|---|---|---|
| Image classifier | MobileNetV2 transfer learning (frozen base), class weights, augmentation | 80.8% test accuracy on 9 classes (480 test images), macro F1 0.806 |
| Text classifier | TF-IDF unigrams and bigrams + logistic regression (C=10) | 99.8% test accuracy (493 / 494); 91.1% when one word per description has a swapped-letter typo |
| Instruction generator | MiniLM retrieval over 76 policy chunks + GPT-2 (124M) fine-tuned on 144 constructed pairs | Share of generated items found in the retrieved text: 0.96, vs. 0.45 for untrained GPT-2 |
| Confidence gate (threshold 0.5) | Applied before generation | Images: answers 89% of inputs at 85.5% accuracy (vs. 80.8% with no gate). Text: no effect on in-style input |

## What I learned and what didn't work

- **Fine-tuning the CNN did not help.** Unfreezing the last 30 base layers gave 81.0% test accuracy vs. 80.8%, which is within noise, with worse validation loss and early overfitting, so I kept the simpler frozen model.
- **The image dataset has a possible shortcut.** Floor color differs by class, so test accuracy may overstate performance on real phone photos.
- **The text data is template-generated.** Near-perfect accuracy mostly reflects an easy task (a 305-word vocabulary), and the model is fragile to typos.
- **Retrieval needed a category filter.** Unrestricted semantic search averaged precision@5 of 0.87, failing mostly on Miscellaneous Trash and Plastic.
- **The generator copies rather than writes.** Training targets were assembled from the retrieved documents, so GPT-2 learned to condense and copy. Probes confirmed it follows its context: editing one rule in the retrieved text changed the output accordingly.
- **A fine-tuned GPT-2 is faithful, not necessarily correct.** It reproduced an inconsistency in the source documents, and one Paper output listed acceptable items under "Not accepted". My groundedness metric could not detect this, because it only checks that text appears in the context, not that it sits under the right label.

## Known limitations

- The confidence gate cannot catch confident mistakes (e.g., a Vegetation image predicted as Food Organics at 0.64 produced fluent, well-grounded instructions for the wrong category).
- Miscellaneous Trash is the weak class for images (recall 0.50), partly because it is a catch-all with no coherent visual identity.
- Per-class test metrics are noisy (32–92 test images per class).
- Only one CNN fine-tuning configuration was tried.
- Retrieved chunks can overlap, which produces repeated lines in generated output.
- No human evaluation of the generated instructions.

## Data

Not included in this repository:

- **RealWaste** image dataset (4,752 photos, 9 classes). Download it from its original source and place the class folders in a `RealWaste/` folder next to the notebook.
- `waste_descriptions.csv` and `waste_policy_documents.json`: generated datasets provided by the bootcamp.

## Tech stack

Python, TensorFlow/Keras (CNN), scikit-learn (text classifier), PyTorch + Hugging Face Transformers (GPT-2), sentence-transformers (retrieval), pandas, matplotlib, seaborn. Everything ran on CPU only (WSL Ubuntu): CNN training took about 75 seconds per epoch and GPT-2 fine-tuning about 19 minutes.

Trained model weights are not included; re-running the notebook regenerates them.
