## auditd rules for ONE collection run (placeholders filled by collect_start_p3.sh). Fixed syscall list, x86_64, restricted to the lab user
## and to the experiment daemons' executables. Rules are OR-ed: one rule per (group, filter).
-D
-b 32768
-f 1
-r 0
__RULES__
