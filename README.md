# Horses for Courses: Classical Ciphers as a Testbed for Architectural Inductive Bias Evaluation

> https://durgarsharma.github.io/architectural-inductive-bias/

## Abstract
This study systematically explores the potential of machine learning models to decrypt text without prior knowledge of the cipher key or explicit rules. Focusing on the feasibility of Zero-Shot generalization, the research compares four dominant neural network architectures: MLP, Character CNN, LSTM, and Transformer. The project is executed across four phases, commencing with a 1,000 article English corpus and extending to Hindi, Greek, and constructed languages (testing systematic variations in morphology, syntax, and phonology). The MLP was trained on five ciphers and evaluated on a held-out sixth cipher in Phase 1, establishing the architectural baseline. Experimental results from the MLP show low to moderate accuracy (50%–65%) with negative generalization gaps, conclusively proving that position independent networks lack the representational capacity for robust decryption. In contrast, the subsequent phases demonstrate that LSTMs, with their explicit sequential memory, achieve near-perfect decryption on position-dependent ciphers like Vigenère, while the Transformer surprisingly fails due to over generalization. The research culminates by challenging the best-performing models with modern block ciphers (AES/DES), serving as a negative control. The expected universal failure validates that neural network success on classical ciphers relies solely on exploitable statistical patternsnot true cryptographic deduction. These findings provide a definitive architectural hierarchy and establish the fundamental boundaries of pattern-based learning across diverse linguistic structures.


## Result 1: Recurrent Memory Solves Sequential Ciphers Universally

- LSTM dominance on Vigenère: 99.91% (English), 98.13% (Hindi), 99.56% (Greek) word accuracy
- Script-invariant performance: Standard deviation ≤ 0.13% across all solvable ciphers
- Proof of architectural necessity: Vigenère accuracy jumps from 8.81% (MLP) → 99.91% (LSTM)
- High stability: Near-zero variance (0.00-0.12% Std Dev) confirms reliable sequential memory

## Result 2: Transformers Catastrophically Fail on Simple Sequences

- Vigenère collapse: 9.88% (English), 13.47% (Greek), 39.23% (Hindi) — performs identically to context-agnostic MLP
- CNN superiority: Character CNN achieves 93.06% vs Transformer's 9.88% on same task
- Extreme instability: High variance on simple ciphers (e.g., Caesar min 19.42% despite 99.93% average)
- Global attention mismatch: Cannot focus on local periodic patterns (i mod 6), over-attends entire sequence

## Result 3: Linguistic Complexity Exposes Fundamental Limits

- Morphological collapse: LSTM drops from 99.91% → 4.43% on Vigenère with agglutinative words (10-20 chars)
- Universal failure: All architectures achieve 0-5% word accuracy despite 91% character accuracy
- Phonological volatility: Variable output classes (PHON-10→50) reduce LSTM to 13.10% on Vigenère
- Architecture inversion: CNN (36.10%) outperforms LSTM (13.10%) under classification instability

## Result 4: Negative Controls Prove Pattern Dependency

- AES/DES universal failure: 0.00% word accuracy across all four architectures (MLP, LSTM, CNN, Transformer)
- Character-level noise: ~22-23% character accuracy (above 3.7% random baseline) but zero meaningful decryption
- Architectural irrelevance: LSTM's recurrent memory provides no advantage on modern secure ciphers
- Definitive boundary: Gap between Vigenère success (99.91%) and AES failure (0.00%) confirms neural cryptanalysis requires exploitable statistical patterns

## Highlights

- We evaluate four architectures on 26,000 encrypted articles, revealing LSTM achieves 99.91% on Vigenère while Transformers fail at 9.88%—performing identically to context-agnostic MLPs despite global attention.
- We demonstrate script-invariant sequential learning with LSTM maintaining 98-100% accuracy (≤0.13% variance), but morphological complexity causes universal collapse to 0-5% despite 91% character accuracy.
- We expose classification instability inverting architectural hierarchy: variable output classes reduce LSTM to 13.10% while CNN achieves 36.10%, showing recurrent memory fails under volatile classification spaces.
- We validate pattern dependency through negative controls: AES/DES yield 0.00% word accuracy across all architectures, proving the Vigenère success depends entirely on exploitable statistical patterns absent in secure ciphers.

## Dataset
We construct a multilingual corpus of 26,000 encrypted articles spanning natural languages (English, Hindi, Greek) and synthetic constructed languages testing morphological, syntactic, and phonological complexity. Articles are sourced from Wikipedia and systematically encrypted across all cipher types.

| Language | Alphabet Size | Articles | Characters | Source |
| :--- | :--- | :--- | :--- | :--- |
| English | 38 | 1,000 | 2,542,530 | Wikipedia API |
| Hindi | 46+ | 1,000 | 1,903,287 | Wikipedia API |
| Greek | 24 | 1,000 | 2,520,499 | Wikipedia API |
| Constructed (Morphology) | 26 | 250×5 levels | Variable | English base, rewritten |
| Constructed (Syntax) | 26 | 250×5 orders | Variable | English base, reordered |
| Constructed (Phonology) | 10-50 | 250×8 levels | Variable | English base, remapped |

Each cipher dataset uses 70% training (700 articles), 15% validation (150 articles), 15% test (150 articles).

## Ciphers

| Cipher | Type | Key Structure | Decryption Challenge |
| :--- | :--- | :--- | :--- |
| Caesar | Monoalphabetic | Single shift offset (e.g., shift=3) | Identify fixed integer offset across corpus |
| Atbash | Monoalphabetic | Alphabet reversal (A↔Z, B↔Y) | Learn mirrored 1:1 mapping rule |
| Affine | Monoalphabetic | Two-key linear: E(x)=(ax+b) mod 26 | Identify multiplication factor *a* and shift *b* |
| Vigenère | Polyalphabetic | Repeating keyword (e.g., "CIPHER") | Track periodic key pattern requiring sequential memory |
| Substitution Fixed | Monoalphabetic | Custom 26-character map (constant) | Learn complete arbitrary substitution table |
| Substitution Random | Monoalphabetic | Different random key per article | Generalize across 1,000 unique random mappings |
| AES | Block cipher | 128/256-bit key, pseudorandom output | Negative control: no exploitable patterns |
| DES | Block cipher | 64-bit blocks, 16-round Feistel | Negative control: statistical randomness |

## Models

| Model | Description | 
| :--- | :--- |
| MLP | Context agnostic feedforward classifier (character level frequency matching, no sequential modeling) | 
| LSTM | Recurrent neural network with memory cells (long-term sequential dependency tracking for periodic patterns) |
| Character CNN | Convolutional architecture with multi-kernel filters (local n-gram pattern detection, window sizes 3-7) |
| Transformer | Global self-attention mechanism (parallel position-aware processing, no explicit recurrence) |



