# e2 charter

Round e2 instructions as Bobby delivered them on 2026-10-21 at 19:25 UTC ("new prompts and context dump for all threads!"). The four text files here are the prompt packet, unedited:

| file | role |
|---|---|
| [00-START-HERE.txt](00-START-HERE.txt) | how the packet is used; snapshot it was prepared against |
| [01-coordinator.txt](01-coordinator.txt) | common contract + coordinator / scientific editor |
| [02-dynamics-explorer.txt](02-dynamics-explorer.txt) | common contract + dynamics explorer |
| [03-mechanisms-explorer.txt](03-mechanisms-explorer.txt) | common contract + mechanisms / measurement explorer |
| [04-independent-auditor.txt](04-independent-auditor.txt) | common contract + independent auditor / adversarial researcher |

Provenance: `modest-complexity-e2-prompts.zip`, SHA-256 `ce7bca06cbf7be22842520d4e0403bfa774f494095dd18ce68e9aafa1e19ce2f`, prepared against `swarm/exploration` at `6cf291fca555e95b3ec221c25e63941d25c06ae0` (an ancestor of the e2 base). The text above the `ROLE:` line is identical in all four role files (checked with `diff`).

The accompanying `modest-complexity-context.zip` (SHA-256 `195bb0b7b05a48aadfe2dde9b3179ad0d6c63c5f7dc6cf6335728306e8dd6220`) holds Neel Nanda's glossary and three papers (Varma et al. 2023, Zhong et al. 2023, Yip et al. 2024). Its READ-ME says the files keep their original authorship and licensing and that inclusion grants no redistribution rights, so only its [READ-ME.md](../../context/READ-ME.md) and [SOURCES.json](../../context/SOURCES.json) (URLs, sizes, SHA-256) are committed. Sessions in this project read the zip from the project files folder (`uploads/hearth/4611f976-…`); others fetch the sources from the URLs in SOURCES.json and check the hashes.

Assignments made under this charter (base commit, owners, run-ID claims) are in [../DECISIONS.md](../DECISIONS.md).
