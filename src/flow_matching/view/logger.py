class Message:
    def __init__(self, msg_id, content):
        super().__init__()
        self.msg_id = msg_id
        self.content = content

    def id(self):
        return self.msg_id

    def read(self):
        return self.content

class Logger:
    def __init__(self, verbose):
        super().__init__()
        self.messages = []
        self.next_id = 0
        self.verbose = verbose

    def log_training_start(self):
        content = "Starting training"
        self._create_msg(content)

    def log_training_end(self):
        content = "Finishing training"
        self._create_msg(content)

    def log_reflow_iteration(self, iteration):
        content = "Reflow iteration: {}".format(iteration)
        self._create_msg(content)

    def log_epoch(self, num_epoch):
        content = "Starting epoch {}".format(num_epoch)
        self._create_msg(content)

    def log_epoch_train_loss(self, loss):
        content = "Training loss: {}".format(loss)
        self._create_msg(content)

    def log_epoch_validation_loss(self, loss):
        content = "Validation loss: {}".format(loss)
        self._create_msg(content)

    def log_device_and_params(self, device, num_params):
        content = "Device: {}, Learnable params: {}".format(device, num_params)
        self._create_msg(content)

    def _get_next_id(self):
        next_id = self.next_id
        self.next_id += 1
        return next_id

    def _create_msg(self, content):
        msg = Message(
            self._get_next_id(),
            content,
        )
        self.messages.append(msg)
        if self.verbose:
            print(content)