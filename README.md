# FinBERT: Financial Sentiment Analysis with BERT

FinBERT is a pre-trained NLP model to analyze sentiment of financial text. It is built by further training
 the [BERT](https://arxiv.org/pdf/1810.04805.pdf) language model in the finance domain, using a large financial corpus and thereby fine-tuning
  it for financial sentiment classification. For the details, please see 
  [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/pdf/1908.10063.pdf).

The model is available on Hugging Face: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert).

---

## Sentiment Pipeline (local LLM via LM Studio)

A one-command pipeline that fetches financial news for a configurable stock universe and runs sentiment analysis through a local LLM served by [LM Studio](https://lmstudio.ai).

### Prerequisites

- Python 3.8+
- LM Studio installed and running with a model loaded on port `1234`
- A free [Finnhub](https://finnhub.io/register) API key

### Setup

**1. Install pipeline dependencies**
```bash
pip install -r requirements-pipeline.txt
```

**2. Configure environment**
```bash
cp .env.example .env
```
Then open `.env` and fill in your Finnhub API key:
```
FINNHUB_API_KEY=your_key_here
```
All other values have sensible defaults and can be left as-is unless you need to change them.

**3. Edit your stock universe**

Open `stocks.yaml` and add or remove tickers:
```yaml
stocks:
  - ticker: AAPL
    name: Apple
  - ticker: MSFT
    name: Microsoft
  - ticker: TSLA
    name: Tesla

settings:
  days_lookback: 7
```

### Running

```bash
# Run with stocks.yaml defaults
python run_sentiment.py

# Override the lookback window
python run_sentiment.py --days 14

# Analyse specific tickers (bypasses stocks.yaml)
python run_sentiment.py --tickers NVDA AMD GOOGL

# Use a different config file
python run_sentiment.py --config my_portfolio.yaml
```

### Output

Results are saved to `output/YYYY-MM-DD/sentiment_HHMMSS.csv` with the following columns:

| Column | Description |
|---|---|
| `run_ts` | UTC timestamp of the pipeline run |
| `ticker` | Stock symbol |
| `news_id` | Finnhub article ID |
| `news_date` | Article publication date |
| `source` | News source |
| `headline` | Article headline |
| `sentiment` | `Positive`, `Negative`, or `Neutral` |
| `confidence_score` | Model confidence (0–1) |

A summary table is also printed to the console at the end of each run.

### Time-series DB push (coming soon)

When your DB endpoint is ready, add it to `.env`:
```
DB_ENDPOINT=http://your-server/api/sentiment
```
Or pass it on the command line:
```bash
python run_sentiment.py --push-db http://your-server/api/sentiment
```

### Environment variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `FINNHUB_API_KEY` | — | **Required.** Your Finnhub API key |
| `FINNHUB_BASE_URL` | `https://finnhub.io/api/v1` | Finnhub base URL |
| `LM_STUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio local server URL |
| `LM_STUDIO_MODEL` | `qwen` | Model name hint passed to LM Studio |
| `OUTPUT_DIR` | `output` | Root folder for CSV results |
| `DB_ENDPOINT` | — | Optional. Time-series DB push endpoint |

---

## Original FinBERT

### Installing
Install the dependencies by creating the Conda environment `finbert` from the given `environment.yml` file and
activating it.
```bash
conda env create -f environment.yml
conda activate finbert
```

## Models
FinBERT sentiment analysis model is now available on Hugging Face model hub. You can get the model [here](https://huggingface.co/ProsusAI/finbert). 

Or, you can download the models from the links below:
* [Language model trained on TRC2](https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/language-model/pytorch_model.bin)
* [Sentiment analysis model trained on Financial PhraseBank](https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/finbert-sentiment/pytorch_model.bin)

For both of these model, the workflow should be like this:
* Create a directory for the model. For example: `models/sentiment/<model directory name>`
* Download the model and put it into the directory you just created.
* Put a copy of `config.json` in this same directory. 
* Call the model with `.from_pretrained(<model directory name>)`

## Datasets
There are two datasets used for FinBERT. The language model further training is done on a subset of Reuters TRC2 
dataset. This dataset is not public, but researchers can apply for access 
[here](https://trec.nist.gov/data/reuters/reuters.html).

For the sentiment analysis, we used Financial PhraseBank from [Malo et al. (2014)](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts).
 The dataset can be downloaded from this [link](https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip?origin=publication_list).
 If you want to train the model on the same dataset, after downloading it, you should create three files under the 
 `data/sentiment_data` folder as `train.csv`, `validation.csv`, `test.csv`. 
To create these files, do the following steps:
- Download the Financial PhraseBank from the above link.
- Get the path of `Sentences_50Agree.txt` file in the `FinancialPhraseBank-v1.0` zip.
- Run the [datasets script](scripts/datasets.py):
```python scripts/datasets.py --data_path <path to Sentences_50Agree.txt>```

## Training the model
Training is done in `finbert_training.ipynb` notebook. The trained model will
 be saved to `models/classifier_model/finbert-sentiment`. You can find the training parameters in the notebook as follows:
```python
config = Config(   data_dir=cl_data_path,
                   bert_model=bertmodel,
                   num_train_epochs=4.0,
                   model_dir=cl_path,
                   max_seq_length = 64,
                   train_batch_size = 32,
                   learning_rate = 2e-5,
                   output_mode='classification',
                   warm_up_proportion=0.2,
                   local_rank=-1,
                   discriminate=True,
                   gradual_unfreeze=True )
```
The last two parameters `discriminate` and `gradual_unfreeze` determine whether to apply the corresponding technique 
against catastrophic forgetting.

## Getting predictions
We provide a script to quickly get sentiment predictions using FinBERT. Given a .txt file, `predict.py` produces a .csv file including the sentences in the text, corresponding softmax probabilities for three labels, actual prediction and sentiment score (which is calculated with: probability of positive - probability of negative).

Here's an example with the provided example text: `test.txt`. From the command line, simply run:
```bash
python predict.py --text_path test.txt --output_dir output/ --model_path models/classifier_model/finbert-sentiment
```
## Disclaimer
This is not an official Prosus product. It is the outcome of an intern research project in Prosus AI team.
### About Prosus 
Prosus is a global consumer internet group and one of the largest technology investors in the world. Operating and
 investing globally in markets with long-term growth potential, Prosus builds leading consumer internet companies that empower people and enrich communities.
For more information, please visit [www.prosus.com](www.prosus.com).

## Contact information
Please contact Dogu Araci `dogu.araci[at]prosus[dot]com` and Zulkuf Genc `zulkuf.genc[at]prosus[dot]com` about
 any FinBERT related issues and questions.
