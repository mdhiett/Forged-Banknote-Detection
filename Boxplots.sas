data _null_;
call symput ('timenow', " " || put (time(), time.));
call symput ('datenow', " " || put (date(), date.));
run;

proc sgplot data=work.import;
  vbox V4 / category=Class;
  title justify=center height=16pt "V4 Box Plot";
  footnote justify=left height=10pt "The current time is &timenow and the date is &datenow";
run;
