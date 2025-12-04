import pandas as pd
from steam_content_based_recsys import SteamContentBasedRecSys, ContentBasedRecommendation, TfidfSteamDatasetVectorizer


if __name__ == "__main__":
    data = pd.read_csv("../combined_steam_games.csv")
    
    recommender = SteamContentBasedRecSys().fit(TfidfSteamDatasetVectorizer, data)

    liked_games = [20200, 655370, 1732930, 1355720, 1139950]

    recommendations = recommender.recommend(liked_games, M=5)

    for sample in recommendations:
        print(f"Игра с id: {sample.recommendation} похожа на игру {sample.original} с коэффициентом = {sample.similarity}")