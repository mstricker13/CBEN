from esa_snappy import GPF

# https://forum.step.esa.int/t/where-to-find-operator-parameters-for-snappy/4621/16
# https://senbox.atlassian.net/wiki/spaces/SNAP/pages/19300362/How+to+use+the+SNAP+API+from+Python


def print_all_gbf_ops():
    op_spi_it = GPF.getDefaultInstance().getOperatorSpiRegistry().getOperatorSpis().iterator()
    while op_spi_it.hasNext():
            op_spi = op_spi_it.next()
            print("op_spi: ", op_spi.getOperatorAlias())

def op_help(op):
        op_spi = GPF.getDefaultInstance().getOperatorSpiRegistry().getOperatorSpi(op)
        print('Op name: {}'.format(op_spi.getOperatorDescriptor().getName()))
        print('Op alias: {}\n'.format(op_spi.getOperatorDescriptor().getAlias()))
        print('PARAMETERS:\n')
        param_Desc = op_spi.getOperatorDescriptor().getParameterDescriptors()
        for param in param_Desc:
            print('{}: {}\nDefault Value: {}\n'.format(param.getName(),param.getDescription(),param.getDefaultValue()))

def print_product_meta_info(product):
    # Input is product as read by ProductIO.readProduct(file_path)
    name = product.getName()
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    band_names = list(product.getBandNames())
    geoScene = product.getSceneGeoCoding().toString()
    print(name, width, height)
    print(band_names)
    print(geoScene)
    # all product methods: https://step.esa.int/docs/v10.0/apidoc/engine/org/esa/snap/core/datamodel/Product.html
    #band = product.getBand('Gamma0_VH') would give values for product
    #vvw = band.getRasterWidth()
    #vvh = band.getRasterHeight()