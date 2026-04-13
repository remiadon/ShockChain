```mermaid
flowchart TD
	node1["extract@news_headlines"]
	node2["extract@sp500_targets"]
	node3["populate_lancedb"]
	node4["train"]
	node1-->node3
	node1-->node4
	node2-->node4
	node4-->node3
```
