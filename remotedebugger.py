def attachDebugger(parser):
    parser.add_argument('--debug',action='store_true',help='Wait for remote debugger to attach')
    a = parser.parse_args()

    if a.debug:
        import ptvsd
        print("Waiting for remote debugger...")
        # Allow other computers to attach to ptvsd at this IP address and port, using the secret
        ptvsd.enable_attach(address = ('0.0.0.0', 3000))
        # Pause the program until a remote debugger is attached
        ptvsd.wait_for_attach()
        #print("Remote debugger connected: resuming execution.")