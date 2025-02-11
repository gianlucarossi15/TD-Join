# TD-Join: Leveraging Temporal Patterns in Subsequence Joins
This Website presents TD-Join, a Matrix Profile based tool for discovering recurrent temporal subsequences by enhacing Matrix Profile with temporal reasoning given by Allen's relations.

## Video
{% include youtube.html id="fHMZUWnqYyQ" %}

[//]: # ([![IMAGE ALT TEXT HERE]&#40;https://img.youtube.com/vi/fHMZUWnqYyQ/0.jpg&#41;]&#40;https://www.youtube.com/watch?v=fHMZUWnqYyQ&#41;)
## Code
The code is available at [https://github.com/gianlucarossi15/TD-Join/](https://github.com/gianlucarossi15/TD-Join/tree/main)
## Time Dependent Matrix Profile
![TDMP](/images/TimeDepedentMatrixProfile.png)
Each column represents the pair of subsequences adhering to the column's name. By perfoming the minimum operation in each column (for any Allen's relation) we retain the best pair of subsequences for each Allen's relation.
## Allen's relation
![allen](/images/allens.png)
In red there are the Allen's relations used to construct the Time Dependent Matrix Profile.

## Architecture
![architecture](/images/systemArchitecture.png)

Time series are stored in the time-series database InfluxDB. Our TD-Join function, used for performing subsequence joins based on the Time Dependent Matrix Profile along with Allen's relations, constitutes the business layer. 

The presentation layer is dedicated to displaying recurrent temporal patterns.
