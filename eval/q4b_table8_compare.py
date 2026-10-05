import json, os
Q=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results/oppu_rep_q4b_2026-09-24','scores')
T=[("news_categorize","oppu_rep",49,123),("news_headline","oppu_rep",100,67),("movie_tagging","oppu_rep",100,33),
   ("citation","oppu_rep_fixed",100,1.2),("product_rating","oppu_rep",101,1.1),("tweet_paraphrase","oppu_rep",100,1.1),("scholarly_title","oppu_rep",101,1.1)]
def hm(d):
    for k in ("accuracy","rouge_1","MAE"):
        if "task_"+k in d: return k
print("%-17s %-8s | %-39s | %-39s" % ("", "", "bf16 (published Table 8)", "4-bit (phone's weights)"))
print("%-17s %-8s | %7s %7s %8s %8s %6s | %7s %7s %8s %8s %6s" % ("task","metric","Task","+User","D","D_u","p_u","Task","+User","D","D_u","p_u"))
for t,r,nu,q in T:
    a=json.load(open(os.path.join(Q,f"score_{t}_r5.json"))); b=json.load(open(os.path.join(Q,f"score_{t}_r5q4.json")))
    m=hm(a)
    f=lambda d: "%7.3f %7.3f %+8.3f %+8.3f %6.3g" % (d["task_"+m], d["oppu_"+m], d["query_level_mean_diff"], d["user_grouped_mean_diff"], d["user_grouped_t_p"])
    print("%-17s %-8s | %s | %s" % (t, m, f(a), f(b)))
print("\nD, D_u oriented so + = better (MAE sign flipped). p_u: user-grouped paired t-test.")
