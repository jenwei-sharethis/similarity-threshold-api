import openai
import json
import boto3
import botocore
import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import roc_curve
import similarity_check_api as sc

def s3_read_file(bucket, key):
    try:
        s3Client = boto3.resource('s3')
        obj = s3Client.Object(bucket, key)
        body = obj.get()['Body'].read().decode('utf-8')
        return body
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"The object {key} does not exist in {bucket}.")
            raise Exception(f"The object {key} does not exist in {bucket}.")
        else:
            print(f"Error reading file from S3: {e}")
            raise e

def s3_upload(bucket, data, folder, timestamp):
    s3Client = boto3.client('s3')
    try:
        s3Client.put_object(
            Body=data,
            Bucket=bucket,
            Key=f'contextual_similarity/ChatGPT-threshold/GPT-validation/{folder}/result-{timestamp}.json'
        )
    except botocore.exceptions.ClientError as error:
        print(f"Error uploading file to S3: {error}")
        raise

class GPTSimilarityThreshold():
    """
    Provide seed articles and candidate articles from contextual similarity.
    Return a (list of) recommended threshold on cosine similarity score.
    Save ChatGPT's justifications on sample of candidate articles to output location.
    """
    def __init__(self, seed_articles, candidate_articles, output_location, sample_size = 100, tpr = [0.5, 0.7, 0.9]):
        """
        :param seed_articles: (dict) json format list of seed articles, keys = ["url", "analyzed_text", "score"]
        :param candidate_articles: (dict) json format list of candidate articles, keys = ["url", "analyzed_text", "score"]
        :param output_location: (list) where to output relevancy results, [bucket, project]
        :param sample_size: (int) number of candidate articles to be sampled for ChatGPT justification
        :param tpr: (list(int)) true positive rate for recommended similarity score thresholds
        """
        self.seed_articles_pd = pd.json_normalize(seed_articles)
        self.candidate_articles_pd = pd.json_normalize(candidate_articles)
        if len(output_location) == 2:
            self.output_location = output_location
        else:
            raise IndexError("Please provide output_location in the following format: [{bucket}, {key}]")

        self.num_candidate_articles = len(candidate_articles)
        self.sample_size = sample_size
        self.tpr = np.array(tpr).sort()

        self._sortSimilarityScore()
        self._sampleCandidateArticles()


    def _sortSimilarityScore(self):
        """
        Sort both seed article table and candidate article table by similarity score
        :return: None
        """
        self.seed_articles_pd = self.seed_articles_pd.sort_values("score", ascending = False)
        self.candidate_articles_pd = self.candidate_articles_pd.sort_values("score", ascending = False)

    def _sampleCandidateArticles(self):
        """
        Sample N data points from candidate article table, which N = sample_size, for later validation
        If number of candidate articles is less than sample size, then we validate all of them
        :return: None
        """
        if self.num_candidate_articles > self.sample_size:
            sample_idx = np.random.choice(self.num_candidate_articles, self.sample_size, replace = False)
            sample_candidates = self.candidate_articles_pd.iloc[sample_idx]
            self.sample_candidate_articles_pd = sample_candidates
        else:
            self.sample_candidate_articles_pd = self.candidate_articles_pd

    def calculateThreshold(self):
        """
        Call GPTSimilarityCheck API to validate a subset of candidate article via ChatGPT.
        Save GPTSimilarityCheck API results to designed output as json file
        Call _ROCCurveThreshold() to calculate suggestive similarity score thresholds based on ROC curve and desired true positive rate from the candidate subset.
        :return: (dict) dictionary of accuracy, suggestive similarity score thresholds for same event and relevant cases
        """

        similarity_check = sc.GPTSimilarityCheck(
                            self._pandas2JsonList(self.seed_articles_pd)
                            )
        GPTresults = pd.json_normalize(similarity_check.candidateRelevancy(
            self._pandas2JsonList(self.sample_candidate_articles_pd)
                                    )
                                       )
        # join on url
        sample_article_results = pd.merge(self.sample_candidate_articles_pd[["url", "score"]], GPTresults, on = "url")
        # save
        self._saveGPTResult(sample_article_results.to_json(orient='records'))
        # return thresholds
        return self._ROCCurveThreshold(sample_article_results)

    def _saveGPTResult(self, data):
        """
        Save GPT results in json format to s3 bucket.
        Returns:
            None
        """
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime('%Y-%m-%d-%H:%M:%S')
        s3_upload(self.output_location[0], data, self.output_location[0], timestamp)

    def _ROCCurveThreshold(self, validate_results):
        """
        Call _relevantThresholds() to calculate suggestive similarity score thresholds for same event and relevant cases.
        :param validate_results: (dataframe) similarity check results done by GPTSimilarityCheck API
        :return: (dict) dictionary of accuracy, suggestive similarity score thresholds for same events and relevant cases.
        """

        # y = response from GPT (same_event, relevant)
        # x = similarity score
        # check edge case

        x = validate_results['score']
        y_same_event = validate_results['same_event']
        same_event_thresholds = self._relevantThresholds(y_same_event, x)

        y_relevant = validate_results['relevant']
        relevant_thresholds = self._relevantThresholds(y_relevant, x)

        threshold_results = {
            "accuracy": self.tpr,
            "same_event": same_event_thresholds,
            "relevant": relevant_thresholds
        }

        return threshold_results

    def _relevantThresholds(self, y_true, y_score):
        """
        Identify edge cases or construct ROC curve based on ChatGPT validation results
        Call _findThresholds() to identify similarity score node that is the closest to desired true positive rate
        :param y_true: (numpy array) contains 0 or 1, if an article is relevant to seeds
        :param y_score: (numpy arrau) similarity score based on embedding cosine similarity
        :return: (list(float)) threshold scores corresponding to desired true positive rate list
        """
        # edge case 1: all positive
        if len(y_true) == len(y_true[y_true == 1]):
            return [np.min(y_score) for i in range(len(self.tpr))]
        # edge case 2: all negative
        if len(y_true) == len(y_true[y_true == 0]):
            print("All samples from candidate data are not relevant.")
            return [np.max(y_score) for i in range(len(self.tpr))]
        else:
            fpr_val, tpr_val, thresholds_val = roc_curve(y_true, y_score)
            return self._findThresholds(tpr_val, thresholds_val)


    def _findThresholds(self, tpr_val, thresholds_val):
        """
        Find the closet node on similarity score for desired true positive rate
        :param tpr_val: (numpy array) true positive rate data points from ROC curve
        :param thresholds_val: (numpy array) similarity score data points from ROC curve
        :return: (list(float)) threshold scores corresponding to desired true positive rate list
        """
        thresholds = []
        for i in range(len(self.tpr)):
            desired_tpr = self.tpr[i]
            idx = np.argmin(np.abs(tpr_val - desired_tpr))
            thresholds.append(thresholds_val[idx])

        return thresholds

    def _pandas2JsonList(self, df):
        """
        Turn pandas dataframe into a list of json objects
        :param df: pandas dataframe
        :return: (list) list of json object
        """
        json_object = df.to_json(orient = 'records')
        return json.loads(json_object)