import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', metavar="MODE", choices=['t', 'c', 'd'],
                        help='mode in which to run compressorinator: t for train, c for compress, d for decompress')
    parser.add_argument('-s', '--srcfile', metavar="SOURCE", type=str,
                        help="name of the file to process... needed for modes c or d")
    parser.add_argument('-d', '--destfile', metavar="DESTINATION", type=str,
                        help="name of the file to output to... needed for modes c or d")
    args = parser.parse_args()

    print(args.mode, args.srcfile, args.destfile)

    if args.mode == 't':
        # Training mode
        import trainer

        trainer.train()
    elif args.mode == 'c':
        assert args.srcfile is not None and args.destfile is not None
        # Compression mode
        import encoder_decoder
        encoder_decoder.encodeImage(args.srcfile, args.destfile)
    else:
        assert args.srcfile is not None
        # Decompression mode
        import encoder_decoder
        helper = encoder_decoder.decodeCompressed(args.srcfile)
        helper.image.show()

        if args.destfile is not None:
            helper.image.save(args.destfile)
