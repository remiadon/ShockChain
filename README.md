```mermaid
flowchart TD
	node1["extract@classif_targets"]
	node2["extract@news_headlines"]
	node3["find_closests_scenarios"]
	node4["populate_lancedb"]
	node5["train"]
	node1-->node5
	node2-->node4
	node2-->node5
	node4-->node3
	node5-->node3
	node5-->node4
```
