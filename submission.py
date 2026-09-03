#Import

import pandas as pd
import zipfile
import os
"""
Export recommendation lists to a 'cvs.zip' file
"""


def export_results(recommendations, user_ids, path="submission.csv", topn=20):
    """
    Write recommendations to a CSV in Codabench's user_id,item_id format,
    then zip it
    """
    rows = []
    for user_id, rec_list in zip(user_ids, recommendations):
        for item_id in rec_list[:topn]:
            rows.append({"user_id": user_id, "item_id": item_id})

    submission_df = pd.DataFrame(rows)

    n_users_covered = submission_df["user_id"].nunique()
    avg_recs_per_user = submission_df.groupby("user_id").size().mean()
    print(f"Submission covers {n_users_covered}/{len(user_ids)} users, "
          f"avg {avg_recs_per_user:.1f} recommendations per user")

    submission_df.to_csv(path, index=False)
    print(f"Saved {path}")

    zip_path = path + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(path, arcname=os.path.basename(path))
    print(f"Zipped to {zip_path} — upload this file to Codabench")

    return submission_df