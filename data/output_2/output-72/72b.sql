/*+ SeqScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id < 865;
/*+ IndexScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id < 865;
/*+ BitmapScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id < 865;