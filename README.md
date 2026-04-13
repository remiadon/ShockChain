```mermaid
flowchart TD
	node1["create_data_dir"]
	node2["extract@news_headlines"]
	node3["extract@sp500_targets"]
	node4["populate_lancedb"]
	node5["train"]
	node2-->node4
	node2-->node5
	node3-->node5
	node5-->node4
```
