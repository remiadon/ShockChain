```mermaid
flowchart TD
	node1["extract@news_headlines"]
	node2["extract@sp500_targets"]
	node3["find_closests_scenarios"]
	node4["populate_lancedb"]
	node5["train"]
	node1-->node4
	node1-->node5
	node2-->node5
	node4-->node3
	node5-->node3
	node5-->node4
```
