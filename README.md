# similarity-threshold-api
This API collects seed and candidate articles with similarity score to median of seed embedding, send a subset of candidate articles to ChatGPT for relevancy validation, and return a recommendation similarity score threshold.

## Dependency
gpt-similarity-check-api
https://github.com/jenwei-sharethis/gpt-similarity-check-api

## Initialize GPTSimilarityThreshold object

`param: seed_articles`: a list of json objects that contains keys = ["url", "analyzed_text", "score"], where url should be a normalized url to avoid duplication, analyzed_text is the text content of corresponding url, and score is the embedding similarity score between this url and the median of seed embeddings.

`param: candidate_articles`: a list of json objects that contains keys = ["url", "analyzed_text", "score"]

`param: output_location`: a list in the format of [{Bucket}, {Key}]. Provide bucket and key to s3 location where the validation results from ChatGPT to be stored.

`param: sample_size`: number of candidate urls being sampled to be sent to ChatGPT for relevancy check. Default is 100.

`param: tpr`: a list of desired true positive rate for recommendation on similarity score thresholds

## Obtain recommeded similarity score thresholds
Use `calculateThreshold()` to obtain recommended similarity score thresholds.

`return`: a dictionary contains keys = ["accuracy", "same_event", "relevant"], where `accuracy` shows the list of desired true positive rate, `same_event` and `relevant` gives a list of recommended similarity score thresholds to use for obtaining desired true positive rates.

### "same_event" vs "relevant"
We present two definition of "relevancy":

`same_event`: the candidate article discuss the same event as seed articles (ex. two news articles talking about 2023 New Year fireworks in SF are considered as same event)

`relevant`: the candidate article has common topics but does not discusses the same event as seeds (ex. a news article talking about Tahoe fireworks is  considered as relevant but not as same event)
