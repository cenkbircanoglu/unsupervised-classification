import pandas as pd


def calculate_accuracy(label_true, label_pred):
    df = pd.DataFrame({'label_true': label_true, 'label_pred': label_pred})
    grouped_count_df = df.groupby(['label_true', 'label_pred']).size().to_frame('size').reset_index()
    max_df = grouped_count_df.iloc[grouped_count_df.groupby('label_true')['size'].idxmax()]
    category_mapping = max_df[['label_true', 'label_pred']].values
    acc = max_df['size'].sum() / len(df)
    informational_acc = (acc * max_df['label_pred'].nunique() / len(set(label_true)))

    return acc, informational_acc, category_mapping


if __name__ == '__main__':
    acc, informational_acc, category_mapping = calculate_accuracy([0, 1, 1, 2, 3, 4], [1, 3, 3, 4, 2, 4])
    print(acc, informational_acc, category_mapping)
