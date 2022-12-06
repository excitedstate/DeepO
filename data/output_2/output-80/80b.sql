/*+ SeqScan(t) SeqScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) SeqScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) SeqScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) SeqScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) SeqScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) SeqScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) IndexScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) IndexScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) IndexScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) IndexScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) IndexScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) IndexScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) BitmapScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) BitmapScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) BitmapScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) BitmapScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) BitmapScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ SeqScan(t) BitmapScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) SeqScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) SeqScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) SeqScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) SeqScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) SeqScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) SeqScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) IndexScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) IndexScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) IndexScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) IndexScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) IndexScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) IndexScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) BitmapScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) BitmapScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) BitmapScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) BitmapScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) BitmapScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ IndexScan(t) BitmapScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) SeqScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) SeqScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) SeqScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) SeqScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) SeqScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) SeqScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) IndexScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) IndexScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) IndexScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) IndexScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) IndexScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) IndexScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) BitmapScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) BitmapScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) BitmapScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) BitmapScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) BitmapScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;
/*+ BitmapScan(t) BitmapScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.production_year > 2002;