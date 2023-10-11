import click

from distributed_inference import run_distributed_inference


@click.command()
@click.option('-mt', '--max_tokens', default=100, help='How many new tokens to generate.')
@click.option('-t', '--temperature', default=1, help='The temperature of the sampling.')
@click.option('-tk', '--top_k', default=200, help='The top_k of the sampling.')
@click.option('-d', '--device', default="cpu", help='The device to run on.')
@click.option('-w', '--workers', default=1, type=click.Choice(['1', '2', '3', '4']), help='The number of workers to simulate.')
@click.option('-s', '--start', prompt='Prompt',
              help='The start of the prompt to use for generation.')
def run(max_tokens, temperature, top_k, device, workers, start):
    run_distributed_inference(
        start, max_tokens, temperature, top_k, device, int(workers))


if __name__ == '__main__':
    run()
