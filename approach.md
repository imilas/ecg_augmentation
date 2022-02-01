	# To Do
- [ ] Settle on a dataset
- Approach:
	- [ ] Apply windowing function to ecg
        - [ ] start with 0.1 second windows
    - [ ] transform/compress each window by applying [minirocket](https://github.com/angus924/minirocket) or [wavelets](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01349-x)
		- [ ] would be nice to if we could check to see if we can recreate the data from this compressed form
 
	- [ ] use transformers or LSTMs because they can learn from temporal data
		- [ ] give transformers all windows before making a classification
        - [ ] example of transformer used for classification: https://moody-challenge.physionet.org/2020/papers/107.pdf
	- Advantage:
		- [ ] this way we do not need heartbeat annotations in order to align beats.
		- [ ] how do we augment this approach
            - [ ] we have to ensure all beats are aligned, so we can use smote
            - [ ] if we don't want to align all beats, maybe some other way of making fake data 
			- [ ] Does it improve classification?