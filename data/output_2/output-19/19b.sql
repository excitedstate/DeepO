/*+ SeqScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id = 18559;
/*+ IndexScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id = 18559;
/*+ BitmapScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id = 18559;