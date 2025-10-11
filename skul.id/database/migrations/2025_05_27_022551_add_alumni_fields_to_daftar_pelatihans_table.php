<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('daftar_pelatihans', function (Blueprint $table) {
            $table->string('nik');
            $table->string('tahun_kelulusan');
            $table->string('npsn');
            $table->string('provinsi');
            $table->string('kota');
            $table->string('alamat');
            $table->string('status_saat_ini');
            $table->string('bidang_pekerjaan')->nullable();
            $table->string('sertifikasi_terakhir')->nullable();
            $table->string('kesesuaian_sertifikasi')->nullable();
            $table->string('nama_universitas')->nullable();
            $table->string('jurusan_universitas')->nullable();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('daftar_pelatihans', function (Blueprint $table) {
            //
        });
    }
};
