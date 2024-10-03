import numpy as np
import matplotlib.pyplot as plt
import os
INF = 5


############################################## Discrete Signal Class ##############################################

class DiscreteSignal:

    def __init__(self, INF=5, values=None, start_impulse=None):
        """Initializes the signal in a given range(-INF,INF)."""
        self.INF = INF  # object -> attributes
        self.start_impulse = start_impulse
        if values is None:
            self.values = np.zeros((2 * self.INF + 1), dtype=float)
        else:
            self.values = values

    def set_value_at_time(self, time, value):
        """Sets the value of the signal at a specific time index."""
        if (-self.INF) <= time <= (self.INF):
            self.values[time + self.INF] = value
        else:
            raise IndexError("Time index is out of bound")

    def shift_signal(self, shift):
        """Returns a new DiscreteSignal instance with the shifted signal x[n - shift]."""
        # rolled and trimmed the shifted values
        shifted_values = np.roll(self.values, shift)
        if shift >= 0:
            shifted_values[:shift:] = 0
        else:
            shifted_values[shift::] = 0

        return DiscreteSignal(self.INF, shifted_values)

    def add(self, other):
        """Returns a new DiscreteSignal instance representing the sum of two signals."""
        # added the values of two signal
        if self.INF != other.INF:
            raise ValueError("signals must be in  same range to add up")

        added_values = self.values + other.values

        return DiscreteSignal(self.INF, added_values)

    def multiply(self, other):
        """Returns a new DiscreteSignal instance representing element-wise multiplication."""
        if self.INF != other.INF:
            raise ValueError("signals must be in  same range to add up")
        # multiplied signal
        multiplied_values = self.values * other.values

        return DiscreteSignal(self.INF, multiplied_values)

    def multiply_const_factor(self, scaler):
        """Returns a new DiscreteSignal instance with the signal multiplied by a constant factor."""
        # values multiplied by a const factor
        multiplied_by_const_values = self.values * scaler

        return DiscreteSignal(self.INF, multiplied_by_const_values)


    def plot(self, figsize=(8, 3),y_range=(-1, 4), title=None, x_label='n (Time Index)', y_label='x[n]', saveTo=None):
        """Plots the signal using matplotlib."""
        # create a new figure
        plt.figure(figsize=figsize)
        # set x-axis ticks
        plt.xticks(np.arange(-self.INF, self.INF + 1, 1))
        # set y-axis range
        y_range = (y_range[0], max(np.max(self.values), y_range[1]) + 1)
        plt.ylim(*y_range)
        # plot the signal (X,Y)
        plt.stem(np.arange(-self.INF, self.INF + 1, 1), self.values)

        # set title
        plt.title(title)
        # set x_label
        plt.xlabel(x_label)
        # set y_label
        plt.ylabel(y_label)
        # add grid to the plot
        plt.grid(True)

        # save plotted graph to the path
        if saveTo is not None:
            plt.savefig(saveTo)

        plt.show()

    # Extra helper functions
    def set_values(self, values):
        """Sets multiple values at once."""
        if len(values) == len(self.values):
            self.values = values
        else:
            raise ValueError("Values array must match the length of current signal.")

    def plot_ax(self, ax, figsize=(8, 3), y_range=(-1, 4), title=None, x_label='n (Time Index)', y_label='x[n]',
             saveTo=None):
        """Plots the signal using matplotlib."""
        # set x-axis ticks
        ax.set_xticks(np.arange(-self.INF, self.INF + 1, 1))
        # set y-axis range
        y_range = (y_range[0], max(np.max(self.values), y_range[1]) + 1)
        ax.set_ylim(*y_range)
        # plot the signal (X,Y)
        ax.stem(np.arange(-self.INF, self.INF + 1, 1), self.values, basefmt=' ')

        # set title
        ax.set_title(title)
        # set x_label
        ax.set_xlabel(x_label)
        # set y_label
        ax.set_ylabel(y_label)
        # add grid to the plot
        ax.grid(True)

        # save plotted graph to the path
        if saveTo is not None:
            plt.savefig(saveTo)

        if ax is None:
            plt.show()

    def __str__(self):
        return f"The signal  values are {self.values} in range {- self.INF} to {self.INF}"



############################################## LTI Discrete Class ##############################################
class LtiDiscrete:
    """Initializes the system with the system impulse response"""

    def __init__(self, impulse_response=None):
        self.impulse_response = impulse_response

    """Decomposes the input_signal and returns the unit impulse and their coefficients"""

    def linear_combination_of_impulses(self, input_signal):

        unit_impulses = []
        coefficients = []

        for i, value in enumerate(input_signal.values):
            unit_impulse = DiscreteSignal(INF=(input_signal.INF), start_impulse=(i - input_signal.INF))
            unit_impulse.set_value_at_time(i - input_signal.INF, 1.0)
            unit_impulses.append(unit_impulse)
            coefficients.append(value)
        return unit_impulses, np.array(coefficients)

    """Fin output of a signal inserted in a Linear Time Invariance system"""

    def output(self, input_signal):
        unit_impulses, coefficients = self.linear_combination_of_impulses(input_signal)

        impulse_responses = []
        output_signal = DiscreteSignal(INF=(input_signal.INF))

        for unit_impulse, value in zip(unit_impulses, coefficients):
            shifted_impulse_response = self.impulse_response.shift_signal(unit_impulse.start_impulse)
            impulse_responses.append(shifted_impulse_response)
            scaled_impulse = shifted_impulse_response.multiply_const_factor(value)
            output_signal = output_signal.add(scaled_impulse)

        return output_signal, impulse_responses, coefficients


############################################## Plotting Function ##############################################

def plot_impulses(unit_impulses, sum_of_unit_impulses, suptitle, which_impulse, plot_name="Impulses"):
    num_responses = len(unit_impulses)
    rows = (num_responses + 3) // 3  # Calculate number of rows needed
    cols = 3  # Number of columns

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axs = axs.flatten()

    # Plot the individual impulse responses
    for i, impulse in enumerate(unit_impulses):
        if which_impulse == "input impulse":
            impulse.plot_ax(ax=axs[i], title=fr'$\delta[n  -  ({i - impulse.INF})] \times[{i - impulse.INF}]$',
                            y_range=(-1, 3))
        else:
            impulse.plot_ax(ax=axs[i], title=fr'$h[n - ({i - impulse.INF})] \ast \times[{i - impulse.INF}]$',
                            y_range=(-1, 3))

    # Plot the sum of the impulse responses
    sum_of_unit_impulses.plot_ax(ax=axs[-1], title="Sum", y_range=(-1, 3))

    plt.tight_layout()

    # Add overall title
    fig.suptitle(suptitle, fontsize=14)
    plt.subplots_adjust(top=.9)

    # plt.show()
    # Get the current working directory
    current_dir = os.getcwd()

    # Create the 'plot' directory in the current working directory
    plot_dir = os.path.join(current_dir, "Discrete_Plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Save the figure in the ./plot directory
    fig_path = os.path.join(plot_dir, f"{plot_name}.png")
    plt.savefig(fig_path)
    print(f"Figure saved as: {fig_path}")

    # Close the plot to free up memory
    plt.close()


def show_linear(lti, input_signal):
    # decompose the input signal into unit impulses
    shifted_unit_impulses, coefficients = lti.linear_combination_of_impulses(input_signal)

    # finding the impulse responses of the system by scaling the shifted_unit_impulses response with the coefficients
    unit_impulses = []
    sum_of_unit_impulses = DiscreteSignal(INF)
    for shifted_unit_impulse, coefficient in zip(shifted_unit_impulses, coefficients):
        scaled_impulse_response = shifted_unit_impulse.multiply_const_factor(coefficient)
        unit_impulses.append(scaled_impulse_response)
        sum_of_unit_impulses = sum_of_unit_impulses.add(scaled_impulse_response)

    # plot the impulse responses and the sum of the impulse responses
    plot_impulses(unit_impulses, sum_of_unit_impulses, suptitle="Impulses multiplied by coefficients",
                  which_impulse="input impulse", plot_name="Impulses")


def show_output(lti, input_signal):
    # find the output of the system
    output, shifted_impulse_responses, coefficients = lti.output(input_signal)
    # finding the impulse responses of the system by scaling the shifted__impulses_response with the coefficients
    impulse_responses = []

    for shifted_impulse_response, coefficient in zip(shifted_impulse_responses, coefficients):
        scaled_impulse_response = shifted_impulse_response.multiply_const_factor(coefficient)
        impulse_responses.append(scaled_impulse_response)

    # plot the output and the impulse responses
    plot_impulses(impulse_responses, output, suptitle="Response of Input Signal", which_impulse="impulse response",
                  plot_name="Impulse_Response")

############################################## Main Function ##############################################
def main():
    # impluse response of the system
    impulse_response = DiscreteSignal(INF)
    impulse_response.set_value_at_time(0, 1)
    impulse_response.set_value_at_time(1, 1)
    impulse_response.set_value_at_time(2, 1)
    impulse_response.plot(title=f"Figure 1: Impulse Response of the system,INF= {impulse_response.INF}")

    # input signal to the system
    input_signal = DiscreteSignal(INF)
    input_signal.set_value_at_time(0, .5)
    input_signal.set_value_at_time(1, 2)
    input_signal.plot(title=f"Figure 2: Input Signal,INF= {input_signal.INF}")

    # create an instance of the LtiDiscrete class
    lti = LtiDiscrete(impulse_response)

    # shows the decomposition of the input signal into unit impulses and the sum of the impulse responses
    show_linear(lti, input_signal)
    # shows the output and the impulse responses of the system
    show_output(lti, input_signal)


if __name__ == "__main__":
    main()
