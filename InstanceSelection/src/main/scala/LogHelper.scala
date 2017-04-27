package com.lsh

import org.apache.log4j.Logger

trait LogHelper{
    val loggerName = this.getClass.getName
    @transient val logger = Logger.getLogger(loggerName)
}
