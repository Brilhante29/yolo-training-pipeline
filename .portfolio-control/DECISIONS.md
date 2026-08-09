# Decision Register: #1 yolo-training-pipeline

| Decision | Selected option | Reason | Revisit trigger |
|---|---|---|---|
| Architecture | deterministic pipeline | ordered data, annotation, training, evaluation and artifact transitions dominate | independent infrastructure boundaries appear |
| Interface | CLI | proof produces artifacts and numbers, not an online response | remote job control becomes the claim |
| Model | YOLO26n from architecture | proves training without mutable pretrained-weight download | domain accuracy becomes the benchmark |
| Runtime | pinned CPU Docker | universal local and CI execution | explicit GPU comparison is funded |
| Messaging/cloud | none | no async delivery or AWS behavior exists | a measured integration requires it |
| License | AGPL-3.0-only | native open-source Ultralytics runtime | model runtime or commercial terms change |

Pure dataset, annotation, aggregation and artifact policies remain outside the framework adapter. Serving, registry and orchestration stay in their owning repositories.
