import pandas as pd


def concat_df(df_1, df_2):
    return pd.concat([df_1, df_2], ignore_index=True)


def switch_sentences(sentence_1, sentence_2):
    return sentence_2, sentence_1


def augment_data(df):
    augmented_df = df.copy()
    # sentence_1과 sentence_2 스위치
    augmented_df["sentence_1"], augmented_df["sentence_2"] = zip(
        *df.apply(
            lambda row: switch_sentences(row["sentence_1"], row["sentence_2"]), axis=1
        )
    )
    # 원본 데이터와 증강된 데이터 합치기
    combined_df = concat_df(df, augmented_df)
    return combined_df
