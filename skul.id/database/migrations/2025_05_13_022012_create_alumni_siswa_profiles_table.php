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
        Schema::create('alumni_siswa_profiles', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained('users')->onDelete('cascade');
            $table->string('nama_lengkap');
            $table->string('nik')->unique();
            $table->year('tahun_kelulusan')->nullable();
            $table->string('asal_sekolah');
            $table->string('jurusan_sekolah');
            $table->string('jenis_kelamin');
            $table->date('tanggal_lahir');
            $table->string('npsn');
            $table->string('provinsi');
            $table->string('kota');
            $table->text('alamat');
            $table->string('status_saat_ini');
            $table->string('bidang_pekerjaan')->nullable();
            $table->string('sertifikasi_terakhir')->nullable();
            $table->string('kesesuaian_sertifikasi')->nullable();
            $table->string('nama_universitas')->nullable();
            $table->string('jurusan_universitas')->nullable();
            $table->string('foto_profil');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('alumni_siswa_profiles');
    }
};
