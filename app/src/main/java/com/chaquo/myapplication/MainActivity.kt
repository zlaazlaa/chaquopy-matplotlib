package com.chaquo.myapplication

import android.content.Context
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.PyException
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        val py = Python.getInstance()
        val module = py.getModule("test-lightrag")

        findViewById<Button>(R.id.button).setOnClickListener {
            try {
                // 调用 Python 的 run_main_sync 函数并获取返回值
                android.util.Log.d("MainActivity", "asdasdasdasdas1111111")
                val result = module.callAttr("test_main")?.toString()
                android.util.Log.d("MainActivity", "Python返回: $result")
                Toast.makeText(this, result ?: "无返回内容", Toast.LENGTH_LONG).show()

                currentFocus?.let {
                    (getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager)
                        .hideSoftInputFromWindow(it.windowToken, 0)
                }
            } catch (e: PyException) {
                Toast.makeText(this, e.message, Toast.LENGTH_LONG).show()
                android.util.Log.e("MainActivity", "Python错误: ${e.message}", e)
            }
        }
    }
}