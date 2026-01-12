# Horses for Courses: Classical Ciphers as a Testbed for Architectural Inductive Bias Evaluation

> https://durgarsharma.github.io/codemixed-tweet-translation/

## Abstract
Hindi and English (Hi–En) code mixing is widespread across Indian social media and remains a major challenge for sentiment analysis and cyberbullying detection. Current multilingual models struggle with this type of data, often giving inconsistent predictions and showing lower confidence. In this work, 27,314 tweets directed at six Indian athletes across three sports (2013–2023) to understand how code mixing affects automated sentiment classification. It was found that 12\% of tweets contain Hi–En code mixing, with significant differences across athletes, and that 48.4\% of all tweets are negative, with Wrestling athletes receiving the highest levels of negativity. Compared the state of the art multilingual models (XLM-RoBERTa, mBERT) with an LLM based translation pipeline and found that these models show only fair agreement on code mixed tweets, highlighting the difficulty of the task. Human evaluation on 498 tweets provides gold standard labels and shows that both multilingual models reach only moderate performance. Overall, the results show the limitations of current multilingual systems for processing code mixed text and demonstrate that translation based workflows offer a strong alternative for sentiment analysis in linguistically diverse social media settings.

## Result 1: Multiligual Models Disagree
- Low agreement: Cohen's κ = 0.253 between XLM-RoBERTa and mBERT
- Statistical significance: χ² = 11,761, p < 0.001
- Extreme divergence: XLM-RoBERTa classifies 42.9%, and mBERT classifies 3.8% as neutral
- Reduced confidence: Both models show significantly lower confidence on code-mixed text (p < 0.001)

## Result 2: Human Evaluation Validates Difficulty
- Moderate inter annotator agreement: Cohen's κ = 0.586
- Both models achieve F1 ≈ 0.59 against gold standard
- Neutral failure: mBERT F1 = 0.07 for neutral sentiment

## Result 3: Translation Works Well
- 76.4% sentiment preservation between raw and translated
- Systematic improvements: 11.5% neutral → negative (better detection)
- Minimal distortion: Only 13.3 character average length increases

## Highlights
- We benchmark XLM-RoBERTa and mBERT on 27,314 code-mixed tweets, achieving only fair agreement (κ = 0.253) and F1 = 0.59 against human labels.
- We validate LLM translation with 76.4% sentiment preservation and 11.5% cyberbullying detection improvement, while models show significantly reduced confidence on mixed text (p < 0.001).
- We conduct human evaluation (498 tweets, κ = 0.586) revealing catastrophic neutral classification failure (F1 = 0.07-0.35) and moderate overall performance across all models.
- We quantify sport-specific cyberbullying: Wrestling 50-77% negative, Cricket 19-45% negative, with code-mixed tweets showing higher negativity (43.4% vs. 35.0%).

## Dataset
We collect a dataset of 27,314 tweets directed at Indian athletes spanning a decade (2013-2023). Our dataset covers 6 athletes across 3 sports (Cricket, Boxing, Wrestling).
| Athlete | Sport | Period  | Tweets |
| :--- | :--- | :--- | :--- |
| Virat Kohli | Cricket | May-Oct 2021 | 4,892 |
| Harmanpreet Kaur | Cricket | May-Oct 2017 | 4,156 |
| Vijender Singh | Boxing | Feb-Jul 2013 | 4,234 |
| Sarita Devi | Boxing | Sep 2014-Feb 2015 | 4,421 |
| Sushil Kumar | Wrestling | May-Oct 2021 | 4,567 |
| Sakshi Malik | Wrestling | Jan-Jun 2023 | 5,044 |

Human Annotations: 498 tweets with gold standard sentiment labels from 3 independent annotators (Cohen's κ = 0.586).

## Models
| Model | Description | 
| :--- | :--- |
| XLM-RoBERTa | Direct multilingual sentiment classification (100+ languages, 2.5TB training data) | 
| mBERT | Direct multilingual sentiment classification (104 languages, Wikipedia-trained) |
| GPT-4o | Code-mixed text translation to English (OpenAI, May 2024) |
| Claude 3.5 | Code-mixed text translation to English (Anthropic, June 2024) |

## Example 
*Code Mixed Hi-En:* ”Yaar, kal ka match bohot intense tha, but Virat ne
amazing performance di!”  
*Translated Tweet:* “Man, yesterday’s match was very intense, but Virat gave an amazing performance!”

*Code Switched Hi-En:* ”I can’t believe we lost the game, lekin Virat ne
bohot achha khela.”  
*Translated Tweet:* “I can’t believe we lost the game, but Virat played really well.”

