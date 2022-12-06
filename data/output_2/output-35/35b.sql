/*+ SeqScan(t) SeqScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) SeqScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) SeqScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) SeqScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) SeqScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) SeqScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) IndexScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) IndexScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) IndexScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) IndexScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) IndexScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) IndexScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) BitmapScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) BitmapScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) BitmapScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) BitmapScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) BitmapScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ SeqScan(t) BitmapScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) SeqScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) SeqScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) SeqScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) SeqScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) SeqScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) SeqScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) IndexScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) IndexScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) IndexScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) IndexScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) IndexScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) IndexScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) BitmapScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) BitmapScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) BitmapScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) BitmapScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) BitmapScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ IndexScan(t) BitmapScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) SeqScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) SeqScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) SeqScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) SeqScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) SeqScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) SeqScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) IndexScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) IndexScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) IndexScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) IndexScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) IndexScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) IndexScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) BitmapScan(mi) NestLoop(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) BitmapScan(mi) MergeJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) BitmapScan(mi) HashJoin(t mi) Leading (( t mi )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) BitmapScan(mi) NestLoop(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) BitmapScan(mi) MergeJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;
/*+ BitmapScan(t) BitmapScan(mi) HashJoin(mi t) Leading (( mi t )) */ explain select count(*) from title t, movie_info mi where t.id=mi.movie_id and t.kind_id = 1 and t.production_year = 1993 and mi.info_type_id > 3;