import mysql.connector
conn = mysql.connector.connect(host='localhost', user='root', password='mysql', port=3308)
cursor = conn.cursor(dictionary=True)
cursor.execute('SELECT id, model_name, file_name, true_angle, pred_angle, confidence, votes_0, votes_90, votes_180, votes_270, applied_criteria FROM nltk.test_orient')
rows = cursor.fetchall()
print(f"Записей в БД: {len(rows)}")
for row in rows:
    f = row['file_name'][:50]
    print(f"  {row['id']}. {f} true={row['true_angle']} pred={row['pred_angle']} c={row['confidence']} v=[{row['votes_0']},{row['votes_90']},{row['votes_180']},{row['votes_270']}] criteria={row['applied_criteria']}")
cursor.close()
conn.close()
