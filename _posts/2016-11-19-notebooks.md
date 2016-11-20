---
layout: post
title: The Notebooks
---


<div class="message">
    A nice box.
</div>

Here are some notebooks:

* [Some tutorials from a tensorflow book I have.]({{site.baseurl}}/assets/TFTextbook.html)
* [Miscellaneous tutorials I've gone through online.]({{site.baseurl}}/assets/TensorflowExamples.html)
* [TFLearn tutorials and Early Stopping practice.]({{site.baseurl}}/assets/TFlearnOnly.html)


Embed code-snippet test:

{% highlight python %}

    def on_epoch_end(self, training_state):
        self.accuracies.append(training_state.global_acc)
        if training_state.val_acc is not None and training_state.val_acc < self.acc_thresh:
            raise StopIteration
     
{% endhighlight %}
