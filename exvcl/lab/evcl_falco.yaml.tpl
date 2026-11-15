# Falco configuration for ONE collection run (placeholders are filled by collect_start_p3.sh). Minimal on purpose; missing keys use Falco defaults.
engine:
  kind: modern_ebpf
rules_files:
  - __RULES__
json_output: true
json_include_output_fields_property: true
json_include_tags_property: false
log_level: info
priority: debug
buffered_outputs: false
# evidence, not alerts: never rate-limit the output
outputs:
  rate: 1000000
  max_burst: 1000000
file_output:
  enabled: true
  keep_alive: true
  filename: __OUT__/falco.jsonl
stdout_output:
  enabled: false
