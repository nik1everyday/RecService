import pandas as pd
import yaml
from sklearn.metrics.pairwise import cosine_similarity


def get_similar_users(user_id, user_similarity_matrix, top_k=3):
    similar_users = user_similarity_matrix[user_id].sort_values(
        ascending=False)[1:top_k + 1]
    return similar_users


def calculate_item_relevance(user_id, userknn_df, user_similarity_matrix):
    similar_users = get_similar_users(user_id, user_similarity_matrix)
    user_relevant_items = userknn_df[userknn_df['user_id'].isin(
        similar_users.index)]
    item_relevance = user_relevant_items.groupby('item_id')['score'].mean()\
        .reset_index()
    return item_relevance


def load_data():
    with open('service/config/config.yaml') as stream:
        config = yaml.safe_load(stream)

    interactions_df = pd.read_csv(config['original_data']['interactions'])
    users_df = pd.read_csv(config['original_data']['users'])
    items_df = pd.read_csv(config['original_data']['items'])
    userknn_df = pd.read_csv(config['userknn_model']['offline'])

    user_item_matrix = interactions_df.pivot(index='user_id',
                                             columns='item_id',
                                             values='watched_pct').fillna(0)
    user_similarity_matrix = pd.DataFrame(
        cosine_similarity(user_item_matrix),
        index=user_item_matrix.index,
        columns=user_item_matrix.index)

    return interactions_df, users_df, items_df, userknn_df, user_similarity_matrix
