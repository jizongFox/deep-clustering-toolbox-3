class WeightScheduler:

    def value(self):
        return NotImplementedError

    def step(self):
        return NotImplementedError

    def state_dict(self):
        """Returns the state of the weight_scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): weight_scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    @staticmethod
    def get_weight(**kwargs):
        raise NotImplementedError
