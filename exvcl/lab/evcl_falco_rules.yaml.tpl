# One rule that records every event of a FIXED list for the experiment services (evidence, not detections).
# Scope in the rule is coarse (lab user, or processes named sshd/vsftpd/nginx); the parser narrows it to the experiment service process trees
# with the master pids written to meta.txt, so the management sshd on port 22 does not contaminate the evidence.
- rule: evcl_events
  desc: Record fixed-list syscall events of the experiment services as evidence
  condition: >
    evt.dir = < and
    evt.type in (execve, execveat, open, openat, openat2, creat, connect, accept, accept4, bind, listen,
                 setuid, setgid, setreuid, setregid, setresuid, setresgid, capset, chroot, mount,
                 clone, clone3, fork, vfork) and
    (user.name = __VICTIM__ or proc.name in (sshd, vsftpd, nginx))
  output: >
    evcl evt=%evt.type pid=%proc.pid ppid=%proc.ppid name=%proc.name exe=%proc.exepath user=%user.name
    fd=%fd.name fdtype=%fd.type flags=%evt.arg.flags
  priority: DEBUG
  source: syscall
